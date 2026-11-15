# V5b validation and baseline preflight

Date: 2026-11-15 UTC. Base main:
`6d019c97a8287db29013e0e2f68213f7f9e386bf` (merged PR #3272).
Branch: `feat/stage9-observer-v5b`.

The [technical contract](stage9_observer_materialization_v5b.md) specifies the
separate V5b inputs, actual byte verification, metadata policy, full accounting,
unsupported operations, risks and future `.pyc` requirements. This document
records executed checks and their limits. It does not qualify installation,
runtime, expectation authority or production.

## Mandatory preflight

Remote main was fetched and independently rechecked before implementation and
before submission. It remains the SHA above. All open PRs were inspected using
paginated API results: there were none before this draft. The branch starts
at current main. Codebase Memory tools and CLI were unavailable in this
environment; direct targeted repository source inspection was used instead.

PR #3272's final head is `0add76cb23573eea094ded4338019fb067a51b27`.
Its [CI workflow](https://github.com/Dudzian/Dudzian/actions/runs/43646908794)
has now completed SUCCESS. Native platform results are separate from portable
synthetic agreement tests:

| Baseline check | Observed result |
| --- | --- |
| Native Windows security | 29 native PASS; separate emulated set 191 PASS |
| [Broad Windows suite](https://github.com/Dudzian/Dudzian/actions/runs/43646908794/job/132608059447) | 25845 PASS, 4143 SKIP, 1559 deselected; job SUCCESS |
| [Broad Ubuntu suite](https://github.com/Dudzian/Dudzian/actions/runs/43646908794/job/132608059417) | 30118 PASS, 169 SKIP, 1539 deselected; job SUCCESS |
| [CodeQL](https://github.com/Dudzian/Dudzian/actions/runs/43646908903) | Python and JavaScript/TypeScript jobs SUCCESS |
| [Main Quality and Security](https://github.com/Dudzian/Dudzian/actions/runs/43648261556) | All nine jobs completed: eight SUCCESS, scheduled history-secret job SKIPPED |
| Main native Windows and broad Windows/Ubuntu | SUCCESS |

SKIP and deselection are not PASS. Main's
[CI aggregate](https://github.com/Dudzian/Dudzian/actions/runs/43648261570)
still had its PostgreSQL issuer-durability job running at the last pre-submission
check and retained the frontend crash described below. It is not represented
as a successful aggregate. New-branch hosted checks must be assessed on this
draft's exact HEAD; baseline checks do not validate new branch bytes.

The completed [platform workflow](https://github.com/Dudzian/Dudzian/actions/runs/43646908972)
has known qualification denials, confirmed against prior #3271 logs and current
gate source. Linux lacks live `LINUX_SYSTEMD_INTEGRATION`, `LINUX_CLEAN_INSTALL`
and `LINUX_REBOOT_LIFECYCLE_UPDATE` evidence. Windows lacks
`WINDOWS_REBOOT_RECOVERY`, `WINDOWS_LONG_RUNNING_LIFECYCLE`,
`WINDOWS_UPDATE_RESTART` and `WINDOWS_FAILED_UPDATE_SAFE_BEHAVIOR` evidence.
Its Core OS, evidence-contract and native Windows installation/integration jobs
passed. Required live evidence gates remain blocking; their refusals were not
relabeled as regressions or suppressed. Physical Linux and Stage 10 dispatch
gates were SKIPPED and establish no qualification.

## Baseline defects repaired before extending validation

1. PR #3272's [Quality ratchet](https://github.com/Dudzian/Dudzian/actions/runs/43646908970/job/132606629236)
   reported 8228 PASS, 20 SKIP and one fixture setup error. The complete
   before/after protected-tree comparison differed only by a disappearing
   `.git/objects/maintenance.lock`. The temporary fixture now disables Git
   automatic maintenance and automatic GC before its first commit. The full
   equality assertion and exporter remain unchanged. The exact maintenance
   race was not deterministically reproduced locally; the narrow fixture
   repair removes that asynchronous mutation source. Its full module passed
   all 30 tests. Main's equivalent Quality workflow had passed independently.
2. Main's [Frontend parity runtime proof](https://github.com/Dudzian/Dudzian/actions/runs/43648261570/job/132616101223)
   reported 16307 PASS, 6 SKIP, then a native segmentation fault and exit 139.
   Investigation reproduced surviving `RuntimeServiceGrpc` and
   `RuntimeServiceAiFeed` workers after the bootstrap tests and GC. Test
   finalizers now retain the application and its Qt owners, stop the streams,
   bounded-join the worker handles and tear down the QML engine with explicit
   cleanup. A new native lifecycle regression requires worker termination.
   Targeted native tests passed 15 cases; a separate bootstrap subprocess
   passed three cases and verified zero surviving runtime workers after GC.
   The full UI suite then passed 16308 cases, with six separately reported
   skips, and actual process exit 0 after the summary. The original native
   crash was intermittent and was not deterministically reproduced; this
   verifies removal of the demonstrated leak and the successful full run,
   rather than proving that every possible Qt shutdown defect is eliminated.

These test-fixture repairs are isolated in commit
`865efd9ac815a951e156734adec76a9f1359625e`. No production UI behavior or
qualification gate was changed.

## Local validation

Functional checks used CPython 3.11.16; the full native UI scope used PySide6
6.7.2 with `QT_QPA_PLATFORM=offscreen`. Tool hooks additionally used an isolated
CPython 3.12 environment. This is tool execution, not execution of the
investigated target CPython or generation of its `.pyc` files.

| Check | Result |
| --- | --- |
| New immutable-member adapter tests | 32 PASS |
| New materialization tests | 114 PASS |
| Combined new V5b suite | 146 PASS, 0 SKIP |
| V1 contract, offline/zstd, V3, V4a–V4c, V5a, real-DEB v2/v3 regressions | 2328 PASS, 0 SKIP with actual `gpgv` |
| CI-contract and secret-hygiene selection | 290 PASS, 0 SKIP |
| Protected-tree fixture regression module | 30 PASS |
| Full native offscreen UI process | 16308 PASS, 6 SKIP, exit 0 |
| Ruff existing full workflow scope | PASS |
| Explicit workflow format scope, including both new deployment modules | PASS, 339 files unchanged |
| Stronger new-module Ruff `E,F,I,UP,B,SIM` | PASS |
| Configured mypy Linux | PASS, 405 source files |
| Configured mypy `--platform win32` | PASS, 405 source files |
| Full pre-commit hooks | PASS: Ruff, ruff-format, configured mypy, staged offline Betterleaks |
| Exact `python -m pre_commit run ruff-format --all-files` | PASS, stable exit 0; no formatter changes |
| Branch-inclusive coverage, both new modules | 99% combined; materialization 99%, adapter 98% |
| Diff coverage against `origin/main`, existing threshold 70% | PASS, 99%; 591 changed measured lines, 3 missing |
| Canonical desktop lock `pip-audit` | PASS, no known vulnerabilities |
| Full offline Betterleaks with unchanged reviewed baseline | PASS, no leaks |
| Blocking custom Semgrep rules | PASS, zero findings; five rule fixtures passed |
| Both new modules: maintained plus blocking Semgrep | PASS, 205 applicable rules, zero findings/errors |

The combined coverage metric includes statements and branches. More precisely,
588/591 statements (99.49%) and 92/96 branch outcomes (95.83%) were covered.
The six full-UI skips are existing empty-parameter-set matrix cases in the
Block P closure/build-readiness tests; they remain SKIP rather than PASS.

The initial regression run had 2298 PASS and 30 SKIP because `gpgv` was absent.
Those skips were not counted as successes. An isolated Debian `gpgv` 2.4.7
binary was then downloaded without privileged installation. Its InRelease was
checked using the existing Debian archive keyring and `sqv`; the Packages index
hash/size and DEB hash/size were checked through that signed chain. The DEB is
`gpgv_2.4.7-21+deb13u1+b5_amd64.deb`, 241100 bytes, SHA-256
`104e4c57b98f0b883aa94c68fbe7ec7b48bcdec7c0f9dedf2a51a5054aaeb5ae`.
The existing `STAGE9_TEST_GPGV` test override selected its absolute path. The
final 2328-case regression run exercised those cryptographic cases. This test
tool provisioning grants no Observer authority or target runtime acceptance.

The full maintained Semgrep scan completed with 25 findings in unchanged
sources, six rule timeouts across four unchanged files, and one partial parse
of an existing C probe. These advisory findings/errors are not zero findings
or proof of full repository analysis. The isolated new-module scan completed
without findings or parsing errors. Baseline CodeQL JS extraction also retains
diagnostics on existing QML `.pragma library` JavaScript files despite its
successful workflow; no scanner or safety gate was weakened.

A first working-tree secret scan encountered only an ignored, generated pytest
bytecode cache containing an existing test key literal. That generated cache
was moved outside the repository and the full scan repeated with the unchanged
baseline. No source secret, finding suppression or baseline entry was added.

The required exact formatter hook was repeated after staging settled; a prior
concurrent-index run reported modifications despite an empty source diff.
Only the final stable exit-zero run establishes this gate.

## Reproduction and scope

Run the new cases and branch coverage with:

```sh
python -m pytest -q \
  tests/deployment/test_stage9_observer_member_source_v5b.py \
  tests/deployment/test_stage9_observer_materialization_v5b.py \
  --cov=deployment.stage9_observer_member_source_v5b \
  --cov=deployment.stage9_observer_materialization_v5b \
  --cov-branch --cov-report=xml
diff-cover coverage.xml --compare-branch origin/main --fail-under=70
python -m mypy
python -m mypy --platform win32
python -m pre_commit run --all-files
python -m pre_commit run ruff-format --all-files
QT_QPA_PLATFORM=offscreen python -m pytest tests/ui_pyside -vv
```

New tests construct actual inert raw/gzip DEBs with GNU/USTAR TARs and compare
separately specified image content and metadata. A Linux integration invokes
the public V3 package verifier and public V4a Release/Packages verifier. Portable
byte tests and Win32 type checking do not substitute for native Windows V5b
execution, nor for any live installation/runtime qualification. Fresh hosted
Windows/Ubuntu and CodeQL results for this draft remain separate CI evidence.

The V1 contract, standard parser activation, V3 inventory semantics and V5a
success definition are unchanged. No limits or coverage thresholds increased;
no security exceptions were added. Zstandard capture, `.pyc`, installation,
maintainer scripts and other excluded transformations remain unsupported.
No merge is authorized or performed.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
External authority acceptance = UNVERIFIED
Authenticated time = UNVERIFIED
Durable anti-rollback = BLOCKED
Runtime qualification = BLOCKED
```
