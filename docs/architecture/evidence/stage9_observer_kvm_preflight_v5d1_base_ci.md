# V5d1 base CI preflight

Observed 2026-11-15, 16:44 UTC. Read-only GitHub inspection;
no rerun, dispatch, hardware experiment or production mutation performed.

## Source baseline

`origin/main` is exactly `da6605a24c6cb29c2b2638ad6f2c2bb5eb5b361a`,
the merge of PR #3276. Its branch head is
`4344784739c0f9361ab54818cdda3ed41a945199`.
Codebase Memory is unavailable in this session; relevant workflow and readiness
source files were read directly after tool discovery.

## Current merge revision

| Workflow / job | Run / job | Observed result |
| --- | --- | --- |
| CI (push: main) | 43733597511 | queued; dependency jobs remain queued |
| Native Windows x64 security (no MSI) | 43733597511 / 132871218302 | SUCCESS |
| Full Python tests Windows / Ubuntu | 43733597511 | NOT_TESTED at this observation; not yet scheduled after wheelhouse |
| Quality and Security | 43733597479 | IN_PROGRESS |
| Quality ratchet | 43733597479 / 132871217037 | IN_PROGRESS; Ruff, format, mypy Linux/Win32 and pure critical tests already SUCCESS |
| Semgrep and working-tree secrets | 43733597479 / 132871217060 | SUCCESS |
| Quality ratchet PostgreSQL remainder | 43733597479 / 132871216807 | SUCCESS |
| Style preflight | 43733597500 | SUCCESS |
| CodeQL | 43733597425 | SUCCESS |
| Windows Build | 43733597516 | SUCCESS |

Run URL prefix: `https://github.com/Dudzian/Dudzian/actions/runs/`.
The full post-merge suite is **not green yet**; missing or pending results are
not passed tests. No completed post-merge failing job was found in the paged
current-revision workflow/run-job responses at this observation.

## Branch and PR evidence, distinct from merge revision

On branch head `43447847`, push CI run 43733147964:

- Native Windows x64 security job 132869839997: SUCCESS.
- Full Python tests Windows job 132870491999: SUCCESS;
  `26522 passed, 4150 skipped, 1559 deselected` in 1197.64 seconds.
- Full Python tests Ubuntu job 132870492030: SUCCESS;
  `30802 passed, 169 skipped, 1539 deselected, 3 warnings` in 831.50 seconds.
- PostgreSQL qualification shards and some downstream jobs remain pending;
  these completed broad jobs do not prove the full workflow succeeded.

PR CI run 43733152321 (merge ref differs from final merge SHA) has completed
Ubuntu Python tests but Windows remained queued at observation. Its focused
native Windows x64 security job 132869952225 succeeded.

## Explained existing failure

PR platform-deployment run 43733152334, job 132870789863,
`linux-deployment-integration`, failed in **Enforce recorded Linux integration
evidence**, after all current core jobs succeeded. Exact log result:

```text
LINUX_SYSTEMD_INTEGRATION=MISSING_CURRENT_EVIDENCE [LIVE_LINUX_INTEGRATION]
LINUX_CLEAN_INSTALL=MISSING_CURRENT_EVIDENCE [LIVE_LINUX_INTEGRATION]
LINUX_REBOOT_LIFECYCLE_UPDATE=MISSING_CURRENT_EVIDENCE [LIVE_LINUX_INTEGRATION]
LINUX_PRODUCTION_READY=NOT_READY
```

The same result occurs before V5c3 on V5c2 branch head `35d8d172`, platform
run 43699570490, job 132770262616 (log at 2026-11-15 10:02:42 UTC).
Direct source confirms `.github/workflows/platform-deployment.yml` invokes
`deployment.platform_readiness` with only aggregated core evidence. The
required live gates are defined in `deployment/platform_readiness.py:91`.
`git diff cca8e837..da6605a2 -- .github/workflows/platform-deployment.yml
deployment/platform_readiness.py deployment/platform_evidence.py` is empty.
This is an existing fail-closed missing physical evidence gate, not evidence
of a new V5c3 code regression. Keep it blocked; do not synthesize live evidence
or weaken the gate to obtain a green workflow.

## Required local quality commands

Use the repository `.venv/bin/python` (3.11.17). It already has Ruff 0.14.11,
mypy 1.10.0, pre-commit 4.7.0, pytest and coverage. Semgrep and Betterleaks were
initially absent. Pinned scanner tooling was subsequently prepared exclusively
under untracked `work/`, with no repository dependency changes or privileged
installation: Betterleaks 1.9.0 at `work/tools/betterleaks` after verifying the
release tarball hash, and Semgrep 1.179.0 in a Python 3.12 wheel-only isolated
environment at `work/semgrep-venv/bin/semgrep`. Setup logs remain local under
`work/v5d1-betterleaks-setup.log` and `work/v5d1-semgrep-setup.log`.
Installation/version checks are not scan results. Semgrep requires writable
settings/cache directories; use `XDG_CONFIG_HOME=$PWD/work/semgrep-state`,
`XDG_CACHE_HOME=$PWD/work/semgrep-state/cache`,
`SEMGREP_SETTINGS_FILE=$PWD/work/semgrep-state/settings.yml` and
`SEMGREP_LOG_FILE=$PWD/work/semgrep-state/semgrep.log`.

```bash
.venv/bin/python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui <new-deployment-modules>
.venv/bin/python -m pre_commit run ruff-format --all-files
.venv/bin/python -m ruff format --check <new-deployment-modules>
.venv/bin/python -m mypy --config-file=pyproject.toml
.venv/bin/python -m mypy --platform win32 --config-file=pyproject.toml
.venv/bin/python -m pre_commit run --all-files
betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 --no-banner --no-color --report-format json --report-path work/v5d1-betterleaks.json .
semgrep scan --config .semgrep/cryptohunter.yml --metrics=off --disable-version-check --error bot_core core deployment scripts
semgrep scan --test --config .semgrep/cryptohunter.yml .semgrep/cryptohunter.py --metrics=off --disable-version-check --jobs 1
```

Semgrep community `p/python` / `p/security-audit` is advisory; custom rules and
fixtures are blocking. Betterleaks is pinned to 1.9.0; Linux x64 release tarball
SHA-256 is `f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7`
in `.github/workflows/quality-security.yml:18` and must be verified on install.
New deployment modules must join the existing explicit Ruff/format inventory,
configured mypy files, Observer tests and coverage in quality-security.yml.
The pre-commit ruff-format hook intentionally excludes deployment, so the
explicit deployment format check is additionally necessary.

GitHub connector combined-status returned no classic statuses and its
commit-workflow helper filters to PR events. `gh api --paginate` was therefore
used to inspect post-merge push runs. Connector job and decoded-log readers
provided the exact evidence above (direct gh log redirect was denied by the
artifact storage boundary). Do not infer a green check from unavailable logs.
