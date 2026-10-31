# V3 complete inert package comparison: four real Ubuntu DEBs

Date: **2026-10-31 UTC**. Base:
`4e5044ac1bff8f078cb27659348f0b2e41b3a4ac`.

The new public package-only verifier compared all four complete Ubuntu 24.04
`amd64` CPython `3.12.3-1ubuntu0.17` DEBs with the separately
[frozen V3 declaration](stage9_observer_3a_real_deb_inventory_v3.json).
The complete operation returned `consistency=MATCH`, scope
`INERT_PACKAGE_INVENTORY_ONLY`. This is **SAME_ORIGIN_OBSERVED_COMPARISON_ONLY;
NOT_INDEPENDENT_ACCEPTANCE**, not source authentication or image qualification.

## Measurement and retained results

The [separate observation](stage9_observer_3a_real_deb_observation_v3.md)
records acquisition, old/new expectation comparison and the freeze before the
first production-parser comparison. The reproduction tool only consumes the
frozen declaration and pre-provisioned local DEBs. It has no download or
expectation-update mode and uses no alternate TAR parser for acceptance.

```sh
python -m scripts.verify_stage9_real_deb_v3 --directory /path/to/exact-debs
```

The measured exit code was **0**, meaning complete package comparison only.
The [result JSON](stage9_observer_3a_real_deb_results_v3.json) retains the whole
DEB sizes/hashes and the independent control/data summaries, verified header
profiles, full TAR sizes/hashes, counts and unchanged qualification refusals.
The [run metadata](stage9_observer_3a_real_deb_run_v3.json) binds the command,
base, measured source-file digests, declaration and result bytes. The final
PR commit and its CI must be checked separately; a local run is not a CI pass.

| Package | Whole SHA/size | Control inventory | Data inventory | Regular bytes / metadata / links |
| --- | --- | --- | --- | --- |
| python3.12-minimal | PASS | MATCH, 6 regular / 1 directory | MATCH, 7 regular / 13 directory / 1 link | MATCH |
| libpython3.12-minimal | PASS | MATCH, 6 regular / 1 directory | MATCH, 283 regular / 24 directory / 1 link | MATCH |
| libpython3.12-stdlib | PASS | MATCH, 3 regular / 1 directory | MATCH, 346 regular / 42 directory / 2 links | MATCH |
| libpython3.12-dev | PASS | MATCH, 2 regular / 1 directory | MATCH, 235 regular / 24 directory / 3 links | MATCH |

All eight streams used the declared `gnu-inert-v3` profile and completed EOF,
frame checksums, TAR termination and ar-region checks. The operation consumed
**52,684,800 decoded bytes**, compared **1,002 records** and retained no partial
package success on failure. The record totals include root directories:
888 regular, 107 directory and seven literal symlinks across control and data.
Counters are summaries; every record and regular-payload digest was compared.

No DEB was installed, no member extracted, no symlink created or followed, and
no target interpreter, package binary, import or maintainer script executed.
The absolute `/etc/python3.12/sitecustomize.py` link target remained data.

## Validation and security review

The unchanged V1/V2 and contract suites plus the new V3 suites passed **1,387
tests**. After the final header-profile report field was added, all **267 V3
tests** passed again. Branch-inclusive coverage of the two new modules and
the reproduction script is **100%**, including all 615 changed executable
lines; the unchanged 70% diff gate passes. This is measured test coverage, not
a proof that the test oracle or toolchain is independently correct.

The inspected negative paths include malformed/unsupported headers and tails,
exact member/metadata/target mismatch, missing and extra records of every kind,
namespace conflicts, parser-level mutations after rebinding outer hashes,
truncated streams, padding and EOF, aggregate limits, native errors, descriptor
and directory substitution, frozen-declaration tampering and cleanup failures.
Trap tests check that package link targets cause no host access or execution.

Review found two accounting issues and one new diagnostic cleanup issue before
publication. Declaration record allowances are now reserved before models are
constructed; decoder buffers are charged once. Frozen-declaration cleanup now
attempts every descriptor, preserves an earlier refusal and refuses success if
closing fails. Regression tests cover the corrected paths. No confirmed P0/P1
accepting bypass was established in this inspected scope; that is not a claim
of repository-wide safety.

The mandatory all-files ruff-format hook, repository Ruff and configured mypy
Linux/Win32 (365 source files each) passed. Another 53 CI/security-toolchain
contract tests passed. Blocking Semgrep 1.179.0 found
zero findings with five rules across 1,091 Python targets; all five rule fixtures
passed. Verified Betterleaks 1.9.0 found no source findings with the existing
configuration and baseline. Its initial whole-tree scan found only ignored,
generated private-key fixture bytecode; that disposable file was removed and
the identical scan passed. No source finding or security rule was suppressed.
pip-audit checked 104 canonical desktop dependencies and the exact Zstandard
dependency with zero known vulnerabilities and no skips; local pip-check passed.
These checks do not qualify the native decoder or host.

The maintained Semgrep community advisory was **BLOCKED**: the enforced proxy
returned CONNECT HTTP 403 for `semgrep.dev`. It was not reported as passed.
Codebase Memory was unavailable; targeted source inspection and actual tests
were used instead, without a structural completeness claim.

## Remaining boundaries

The legacy Ubuntu adapter, `ObserverManifestV1`, image closure and source/file
bindings were not widened. V3 creates no archive-to-image mapping or runtime
authority. The GNU profile deliberately refuses every non-NUL extension byte;
PAX, sparse, long-name and other unsupported records still refuse. Declarations
and tools remain unapproved, budgets cooperative, and descriptor rechecks are
not complete TOCTOU protection or a hard process sandbox.

V4 APT provenance and V5 image mapping remain separate next steps. Independent
acceptance, runtime closure, trusted launch, scope/replay authentication and
physical qualification remain open. No Dell test or Stage 3B ran. Enrollment
and native-platform regressions in PR #3228 were not modified by this change.

```text
SOURCE_AUTHENTICITY=UNVERIFIED
ARCHIVE_TO_TARGET_METADATA_MAPPING=NOT_YET_PROVEN
stage_9=IN_PROGRESS
qualification=BLOCKED
CONTRACT_APPROVED=NO
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
runtime_attestation=MISSING
host_pid_namespace=UNVERIFIED
V2_COMPLETE=NO
```
