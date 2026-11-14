# Observer archive-to-image V5a: local validation

Date: 2026-11-14. Base main: `5d56a05192436b8772ab0fb73bbeac0d3ec91a00`.
Branch: `feat/stage9-observer-image-v5a`. The Draft PR description records the
exact delivered HEAD. Main and active PRs were checked before implementation;
the search returned no open PRs. Main was checked again before delivery.

The [technical contract](stage9_observer_image_v5a.md) defines the static
mapping, unsupported transformations, unresolved mappings and V5b boundary.
Historical operator evidence, standard Observer parsers, readiness gates and
V1/V3/V4 semantics are unchanged. Only the existing explicit CI quality scopes
were extended to include the new module/tests; thresholds, previous scopes,
timeouts and opt-in hardware gates were preserved.

## Environment and results

Validation ran as UID 1000 on Linux, CPython 3.12.14. Tooling was provisioned in
a separate workspace venv: Ruff 0.14.11, mypy 1.10.0, pre-commit 4.7.0,
Semgrep 1.179.0 and Betterleaks 1.9.0. The Betterleaks binary was checked against
the repository-pinned SHA-256 before use. No investigated package was installed,
extracted into a system tree, imported or executed.

| Check | Local outcome |
| --- | --- |
| New V5a tests | **119 passed**, no skips or xfails on Linux |
| V5a statement/branch coverage | **99%**, 456 statements, 90 branches, no partial branches |
| V1, V2/V3 and V4a/b/c regression selection | **2145 passed, 30 skipped** |
| CI workflow contracts, seven test modules | **276 passed** |
| Secret-hygiene security tests | **13 passed** |
| Full existing workflow Ruff baseline, with V5a added | PASS |
| Explicit CI format scope, 335 files | PASS |
| Exact `python -m pre_commit run ruff-format --all-files` | PASS, no changes |
| Full `python -m pre_commit run --all-files` | PASS, including staged Betterleaks |
| Configured mypy Linux and `--platform win32` | PASS, 403 source files each |
| Betterleaks complete working tree, unchanged redacted baseline | PASS, zero leaks |
| Semgrep custom blocking rules | PASS, zero findings; rule fixtures **5/5 passed** |
| Semgrep maintained community rules on new V5a module | PASS, 200 rules, zero findings/errors |
| Diff whitespace and scope inspection | PASS |

The 30 existing regression skips all report exactly:
`A preprovisioned local gpgv is required; tests never download a tool`.
No local `gpgv` executable was available. The existing skip policy and tests
were not changed, and no skip/xfail was added to conceal a failure. Portable
V4b policy, signature-adapter denial and V4c cryptographic/contract tests ran;
the absent-tool cases are not claimed as verified real-gpgv execution.

The full advisory Semgrep community scan reported 25 findings in unchanged
existing files, seven rule timeouts across three existing Python files and a
partial parse of an existing C diagnostic. The separate complete V5a community
scan had zero findings/errors. No scanner rule or baseline was weakened.
Generated pytest/cache artifacts that produced secret-scanner fixture matches
were retained outside the repository before repeating the full unchanged scan;
no source or historical evidence was altered to suppress a finding.

Windows portability was checked through configured Win32 mypy and portable
mapper tests. DEB, TAR, Release, Packages and control fixtures use explicit
bytes and LF, avoiding platform newline conversion. Only the actual V3/V4a
descriptor integration has an explicit existing-platform boundary; the static
mapper tests run without Linux system APIs. A native Windows interpreter was
not available in this Linux session; Win32 mypy is not native Windows execution.

Codebase Memory MCP and its CLI were unavailable in this environment. Direct
source inspection and targeted repository searches were used; no graph-index
coverage or freshness claim is made.

## Reproduction

Run from the repository with the pinned quality tools and test dependencies:

```bash
python -m pytest tests/deployment/test_stage9_observer_image_v5a.py -q \
  --cov=deployment.stage9_observer_image_v5a --cov-branch --cov-report=term-missing
python -m pytest tests/test_stage9_observer_contract.py \
  tests/deployment/test_stage9_observer_{offline,zstd,tar_v3,package_v3,apt_v4a,apt_v4b,apt_v4b_crypto,gpgv_v4b,apt_v4c,trust_v4c,checkpoint_v4c}.py \
  tests/deployment/test_stage9_real_deb_{v2,v3}.py -q
python -m pytest \
  tests/scripts/test_ci_workflow_security_scan_contract.py \
  tests/scripts/test_general_ci_platform_contract.py \
  tests/scripts/test_history_reproposal_quality_ci.py \
  tests/scripts/test_linux_physical_evidence_workflow.py \
  tests/scripts/test_platform_core_evidence_workflow.py \
  tests/scripts/test_s9_fv_01_junit_workflow.py \
  tests/scripts/test_stage9_safe_ci.py tests/quality/test_secret_hygiene.py -q
python -m mypy --config-file=pyproject.toml
python -m mypy --config-file=pyproject.toml --platform win32
python -m pre_commit run ruff-format --all-files
python -m pre_commit run --all-files
semgrep scan --config .semgrep/cryptohunter.yml --metrics=off \
  --disable-version-check --error bot_core core deployment scripts
semgrep scan --test --config .semgrep/cryptohunter.yml .semgrep/cryptohunter.py \
  --metrics=off --disable-version-check --jobs 1
betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 \
  --no-banner --no-color .
```

The full Ruff/explicit-format and community-Semgrep selections are the unchanged
commands in `.github/workflows/quality-security.yml`, extended only for V5a.
Keep pytest, coverage, mypy and scanner caches/reports outside the repository
when reproducing security scans. Session logs/JUnit/coverage and scanner JSON
were retained in the workspace, not committed as operator evidence.

## Interpretation

Tests cover independently specified expected content, all direct metadata,
closed trees, explicit identical shared directories, collisions, source/record
identity, traversal/aliases, literal symlink variants, unsupported transforms,
snapshot/declaration substitution, malformed nested evidence, resource limits,
and interruptions both before mapping and after complete staging. A valid
400-file declaration exceeding 64 KiB checks V5a's separate document bound.
Valid V4b/V4c blocker tuples remain intact after a local mapping match.

Local validation is not completion of all remote CI jobs, an authenticated
transcript, authority acceptance, an installed image or runtime qualification.
The PR remains a draft and must not be merged automatically.

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
