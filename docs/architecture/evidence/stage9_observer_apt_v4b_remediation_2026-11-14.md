# PR #3270 — bounded P1/P2 remediation, 2026-11-14

Before editing, GitHub PR metadata, the remote branch and the fetched checkout
all identified reviewed HEAD `c9d1b0b8a380351c9127714e32ddfb49e4dfff19`.
The complete PR diff against `dcc71020c14ff1c09c49cd8be99f7740e490a818`
was inspected, including both V4b modules. A separate worktree preserved the
original local checkout. This report supplements the [original validation
record](stage9_observer_apt_v4b_validation_2026-11-14.md).

## P1: direct public-adapter privilege refusal

`verify_detached_v4b()` now checks `os.geteuid() != 0` immediately after its
Linux platform check, before tool opening, sealed descriptor allocation or
subprocess creation. EUID zero raises the existing controlled
`GpgvRefusalV4b("GPGV_INPUT_REJECTED")`; the refusal-code vocabulary is unchanged.
The existing outer `verify_apt_signature_v4b()` privilege check is retained.

The negative test replaces EUID with zero, makes `os.open` fail the test if
called and asserts that the harness recorded no process launch. Positive
adapter tests cover EUID 1, 1000 and 65534 through the existing real-child
protocol harness. These are simulated UID branches; the entire suite itself
ran as Linux UID 1000. Separate genuine-gpgv cryptographic tests also passed.

This guard does not independently approve a caller-selected ELF, its pin,
dynamic loader, libraries, runtime or kernel. It does not provide an OS
sandbox, protect against a hostile same-runtime caller, or change the fixed
`tool_acceptance=UNVERIFIED` / `system_isolation=UNVERIFIED` claims.

## P2: effective DEB size and boundary evidence

No limit or runtime accounting behavior was changed. `_frozen_archive()`
already copies incrementally into a memfd, but conservatively charges the
complete kernel-backed copy plus one chunk. `_Budget.retained` is cumulative:
deleting metadata and closing an earlier DEB descriptor do not refund it.
The mandatory final input-tree scan also reserves memory.

For memory ceiling `M`, chunk size `C`, retained amount `R` immediately before
a copy and declared DEB size `D`, copy admission requires `D + C <= M - R`,
in addition to `D <= max_file_bytes`. For one raw/gzip DEB, complete memory
admission also requires `D + C + S <= M - R`, where `S` is the final scan's
reservation. Multiple copies accumulate `sum(D_i + C)`; Zstandard has an
additional remaining-workspace check. All other byte, member, format, hash,
sealing, deadline and cleanup checks still apply. The [technical
contract](../stage9_observer_apt_v4b.md) details the preceding reservations,
compressed-index allowances and streaming-model tradeoffs.

The exact single-package plain-index fixture has `R = 1,723,520`,
`S = 3,272`, `C = 65,536`, and default `M = 33,554,432` bytes:

| Boundary | Exact bytes | Public outcome |
| --- | ---: | --- |
| Complete memory ceiling | 31,762,104 | VALID/MATCH local evidence |
| Same valid input, memory ceiling reduced by one byte | unchanged DEB | LIMIT_EXCEEDED; all evidence discarded |
| Same fixture with one extra legal inventory whitespace byte | unchanged DEB | Additional 16-byte charge; LIMIT_EXCEEDED |
| Next valid even ar size | 31,762,106 | Copy succeeds, final scan refuses |
| Copy-admission ceiling | 31,765,376 | Copy succeeds, final scan refuses |
| Copy-admission ceiling plus one byte | 31,765,377 | Refuses before allocating a DEB memfd |
| Declared 32 MiB or 64 MiB | 33,554,432 / 67,108,864 | Memory refusal before DEB memfd |
| Declared max_file_bytes plus one | 67,108,865 | Inventory size refusal |

These sizes are fixture-specific, not a universal 30.29 MiB API constant.
Valid ar lengths are even; the one-byte admission overflow intentionally need
not be a valid archive because it must refuse before allocation or parsing.
A complete two-package test reaches exactly the default cumulative memory
ceiling, then fails with one less budget byte, confirming that closed copies
remain charged. The boundary cases invoke the public API and actual DEB
copy/seal/parse/final-scan path; their existing crypto fixture is mocked.
Separate real-signature complete-chain tests retain genuine cryptography.

## Actual validation

Unprivileged Linux; CPython 3.11.16; pytest 9.1.1. Only inert synthetic DEBs
and disposable/frozen TEST keys were used. No investigated package was
installed or executed. Test `gpgv` 2.4.7 was extracted separately without
system installation from the exact package/hash recorded in the [fixture
README](../../../tests/fixtures/stage9_observer/apt_v4b/README.md).
Its executable SHA-256 is
`3f29dddc10e4089aeac5b2675313f4e5fe822abe5ab3bf7898d238c32d758304`.
This setup provenance is not independent tool acceptance.

| Check | Actual result |
| --- | --- |
| New P1 UID cases | 4 passed; root no-open/no-process and three positive UIDs |
| New P2 public boundary cases | 6 passed |
| All V4b tests | 207 passed, including 11 dedicated genuine-crypto tests and eight genuine signed complete-chain cases |
| Unchanged V1–V4a and V2/V3 reproduction regressions | 1,580 passed |
| Final combined eleven-file invocation | **1,787 passed**, zero skips/failures, 27.67 seconds |
| Existing Stage 9/coverage/workflow contract tests | 181 passed, zero skips/failures |
| V4b branch-inclusive coverage | 98% combined; APT 99%, adapter 97% |
| Changed executable lines versus reviewed HEAD | 100% (one line) |
| Changed executable lines versus origin/main | 98%; unchanged 70% gate passed |
| Workflow's complete Ruff baseline | Passed |
| Strict E/F/B/I Ruff and explicit format check for both modules and all three V4b test files | Passed |
| Exact `python -m pre_commit run ruff-format --all-files` | Passed; no unrelated formatting changes |
| Full staged pre-commit: Ruff, format, configured mypy, offline Betterleaks | Passed |
| Configured mypy Linux and Win32, no filename narrowing | Passed, 399 files each |
| Semgrep CE 1.179.0 configured blocking source scope | Five rules, 1,131 targets, zero findings |
| Semgrep rule fixtures | 5/5 passed |
| Maintained community Semgrep over both V4b modules | 200 applicable rules, zero findings/errors |
| Verified Betterleaks 1.9.0 full working tree and final staged scope | Passed, zero findings; baseline/config unchanged |
| pip-audit canonical desktop lock | 104 dependencies, zero known vulnerabilities/skips |
| pip-audit exact Zstandard 0.25.0 lock | One dependency, zero known vulnerabilities/skips |
| Test environment `pip check`; diff whitespace | Passed |

The combined test invocation selects the same eleven files documented in the
original validation record, with `STAGE9_TEST_GPGV` pointing to the separately
provisioned verifier, both V4b modules in `--cov`, and `--cov-branch`.
JUnit XML and branch coverage XML were retained under the task's local
`work/validation/`. The additional 181 cases are
`tests/scripts/test_stage9_safe_ci.py`,
`tests/scripts/test_history_reproposal_quality_ci.py` and
`tests/scripts/test_linux_physical_evidence_workflow.py`.
Both comparison scopes passed `diff-cover --fail-under=70`.

Semgrep used the existing blocking scope `bot_core core deployment scripts`
and unchanged exclusions/size policy; one oversized file and four ignored
files were outside that scan. No repository-wide community-advisory-clean
claim is made. Betterleaks' archive matched both the workflow's pinned hash
and the release checksum list. Tooling/caches remained outside tracked source;
generated test bytecode/cache artifacts were removed before the working-tree
scan. Initial scanner cache/log writes to the read-only home failed; writable
task-local cache/log paths resolved those setup errors without suppressions.

Validated source SHA-256 values:

```text
89117e623d4b79f9ef52610d81e279a35c517e1c5ecc301f227074dc0429bdd4  deployment/stage9_observer_apt_v4b.py
902473c6636f8f25caf419853cfe229004507b2a76d078662e617a9862071007  deployment/stage9_observer_gpgv_v4b.py
```

Source review confirmed the guard, accounting and test paths; an additional
independent review found no blocking issue in this bounded diff. This is not
an exhaustive security proof. Local validation does not imply successful
remote CI; the PR records the published HEAD and its remote workflow snapshot.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```

V4a, V5, production policies and hardware/manual gates remain unchanged.
The existing Draft PR is updated without merge or automatic merge.
