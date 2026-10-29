# Stage 9 minimal CPython scope — development checks

Development checks: 2026-10-28 UTC; final document review and integrity checks:
2026-10-29 UTC. Repository baseline:
`3946695a8c7a37b09e9c494bb07caf68ccbbf81f`.
Development interpreter: CPython **3.12.14**, selected by
`/workspace/work/stage9-venv/bin/python`; this is not the Dell target build.
Tools: pre-commit **4.5.1**, Ruff **0.14.11**, pytest **9.0.2**,
Betterleaks **1.9.0**. No inspected target ELF, Observer, Stage 3B, physical
procedure or hardware test was executed by these checks.

## Mandatory repository formatting hook

Executed exactly, with the task virtual environment/cache locations selected:

```sh
PATH=/workspace/work/stage9-venv/bin:$PATH PRE_COMMIT_HOME=/workspace/work/pre-commit-cache XDG_CACHE_HOME=/workspace/work/xdg-cache PIP_CACHE_DIR=/workspace/work/pip-cache VIRTUALENV_OVERRIDE_APP_DATA=/workspace/work/virtualenv-app-data python -m pre_commit run ruff-format --all-files
```

Exit **0**: `ruff format..............................................................Passed`.
The hook initialized its configured repositories and installed the Ruff hook.
Network access was explicitly enabled for the command within the configured
proxy and allowed-host policy. No hook scope, formatter configuration,
exclusion or suppression was changed. `git status --short` was empty after
the checks, before the parent added the documentation packet. No production
file was reformatted.

## Tier A Ruff baseline and explicit parser/test format check

```sh
/workspace/work/stage9-venv/bin/python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
/workspace/work/stage9-venv/bin/python -m ruff format --check deployment/stage9_observer_contract.py tests/test_stage9_observer_contract.py
```

Both exit **0**: `All checks passed!`, `2 files already formatted`.
The baseline is the command in `scripts/quality/run_fast.ps1`; PowerShell
itself was unavailable, so its relevant constituent checks were used.
The shared canonicalizer was inspected but not rewritten or broadly formatted.

## Existing rootless synthetic contract tests

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0 /workspace/work/stage9-venv/bin/python -m pytest tests/test_stage9_observer_contract.py -q -m 'not hardware and not destructive and not network'
```

Exit **0**: **869 passed in 2.45s**. The source module states that these are
rootless synthetic contract tests and that no artifacts or host processes
execute. They verify current parser behavior and negative paths, not runtime
closure, independent authority or a target installation. No tests were added
for this documentation-only analysis.

## Offline Betterleaks

The stable Linux x64 1.9.0 asset was fetched from the same release URL as
`.github/workflows/quality-security.yml` and verified both against the
workflow-pinned SHA-256
`f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7`
and the release `checksums.txt`. Both checks exit **0**; the binary reports
`1.9.0`. It resides outside the repository in `/workspace/work/betterleaks`.

```sh
/workspace/work/betterleaks/betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 --no-banner --no-color --report-format json --report-path /workspace/work/betterleaks-working-tree.json .
```

Full local working-tree scan: exit **1**, approximately **80.97 MB** in
**2.04s**, exactly one finding (`private-key`). Its path is the pre-existing
ignored compiled test cache
`tests/__pycache__/test_runtime_bootstrap.cpython-311-pytest-9.1.1.pyc`,
reported line 224. Git confirms the file is untracked and ignored by
`.gitignore:1` (`__pycache__/`); its filesystem mtime is
**2026-10-06 00:26:54 UTC**, predating this task. Source
`tests/test_runtime_bootstrap.py:500` contains the `_RISK_TLS_KEY` fixture.
No secret bytes were printed. This local cache finding does not establish a
new documentation leak, but the complete local scan is **not PASS**.
The cache, baseline, scanner rules and finding were not suppressed or altered.

The full scan above ran before the new documentation files were present.
The final explicit-path hook is recorded separately below.

## Documentation integrity and changed-path scan

The exact retained operator input is **6164 bytes**, SHA-256
`6cee8b1204fa73956aa298a41b951dd34de23c5ea89cc3949fc75d17eb353708`.
It matches the upload byte for byte. This is text retention, not independent
authentication of the reported Dell measurements.

Final checks cover the four new text files and the new document manifest:
UTF-8 decoding, relative file links, balanced fences, table column counts,
trailing whitespace, all eight required status values, complete direct import
statement coverage against static ASTs of the two source files, reference
source hashes/line ranges, and the documentation-only diff. The reference
hashes identify the downloaded upstream bytes, not accepted Ubuntu artifacts.

The three historical manifests verify unchanged: review R1 **22/22**, Dell
local ELF V1 **4/4**, Ubuntu provenance V1 **6/6**. The new manifest has
**19/19** entries and excludes itself. No historical bytes were edited.

```sh
BETTERLEAKS_BIN=/workspace/work/betterleaks/betterleaks /workspace/work/stage9-venv/bin/python scripts/quality/betterleaks_hook.py docs/architecture/stage9_observer_3a_minimal_cpython_runtime_scope_v1.md docs/architecture/evidence/stage9_observer_3a_cpython_semantic_sources_v1.md docs/architecture/evidence/stage9_observer_3a_minimal_cpython_source_v1.txt docs/architecture/evidence/stage9_observer_3a_minimal_cpython_validation_v1.md docs/architecture/evidence/stage9_observer_3a_minimal_cpython_v1.sha256
```

Exit **0**, no leaks found for the final documentation paths. This scoped
result does not turn the separate full-local-tree scan into PASS. Scanner
rules and baseline remain unchanged. `git diff --cached --check` passes;
all five changed files are added documentation/evidence text only.

## Remaining checks / limits

No mypy, full Tier B, Semgrep, dependency audit, coverage ratchet or hosted
exact-HEAD CI result is claimed by this note. Codebase Memory MCP/CLI was
unavailable; no structural-index coverage conclusion is available.
