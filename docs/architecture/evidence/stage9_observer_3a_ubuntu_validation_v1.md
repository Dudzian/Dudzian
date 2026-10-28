# Local validation notes — 2026-10-28 UTC

Repository worktree: `/workspace/work/stage9-provenance`.
Base: `d8a1df2e818c15e151e1e792a684411b5b4dda93`.
Python used: `/workspace/.onboarding/venv/bin/python`, Python 3.11.16.
Ruff: 0.14.11. pytest: 9.1.1.

## Mandatory formatting preflight

Executed exactly (with PATH selecting the existing venv Python):

```sh
PATH=/workspace/.onboarding/venv/bin:$PATH python -m pre_commit run ruff-format --all-files
```

Exit 1: `No module named pre_commit`. The default Python also lacks that module.
No pre-commit package/cache was discovered in the inspected onboarding caches.
Shell network access was already unavailable through the configured proxy;
no proxy bypass or external package installation was attempted.
This is **not a passed mandatory hook**. AGENTS.md requires reporting this gap
and leaving the PR draft until the separate `Style preflight` workflow passes.
That workflow runs the same hook with `--show-diff-on-failure` on all tracked files.

Supplementary read-only formatting check used the hook filename include/exclude
patterns over `git ls-files`, then ran Ruff 0.14.11:

```text
python -m ruff format --force-exclude --check <2416 tracked candidate paths>
2410 files already formatted
```

Exit 0. `--force-exclude` applies pyproject.toml's existing exclusions. An initial
exploratory check without this option also considered excluded generated/archive
files and reported four legacy excluded files; no formatter mutation was used.
The supplementary check does not replace the mandatory unavailable pre-commit hook.

## Tier A lint

```sh
/workspace/.onboarding/venv/bin/python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
```

Exit 0: `All checks passed!`. This is the Ruff command from
`scripts/quality/run_fast.ps1`. PowerShell (`pwsh`) is unavailable.

## Existing synthetic contract regression suite

Inspected `tests/test_stage9_observer_contract.py`: its module scope states
rootless synthetic contract tests; it imports the parser and canonicalizer.
No Observer runtime, launcher, target package binary, host scan or physical
qualification procedure is executed by this suite.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0 /workspace/.onboarding/venv/bin/python -m pytest tests/test_stage9_observer_contract.py -q -m 'not hardware and not destructive and not network'
```

Exit 0: **869 passed in 3.85s**. These are schema regression checks, not
independent laptop measurements or authentication of operator evidence.

## Secret-scan limitation and unchanged files

`betterleaks` is unavailable on PATH, no `BETTERLEAKS_BIN` executable was
selected, and no cached copy was found in inspected onboarding caches.
The required offline Betterleaks gate remains a validation gap for GitHub CI;
do not claim that scan passed. `scripts/quality/betterleaks_hook.py` was inspected:
it refuses a nonempty scan scope when the executable is absent.

No tracked file was modified by these checks. `git diff --stat` was empty at
completion; only the parent's newly added untracked evidence files were visible.
No mypy, full Tier B, hardware, destructive, Stage 3B or physical Dell tests were run.

## Documentation evidence checks

The final documentation checker verifies exact retention of the 9661-byte upload,
all 25 supplied SHA-256 values, all seven package names/versions/hashes and all
twelve ELF rows against the input. The twelve ELF hashes also match the historical
Dell report. It checks the eight mandatory status lines, relative Markdown links,
UTF-8 text, trailing whitespace and balanced fenced blocks. All pass.

Both historical manifests verify unchanged: review R1 **22/22**, Dell V1 **4/4**.
The new V1 manifest verifies **6/6** entries, including both historical manifests;
it excludes itself. The diff contains five added text documentation/evidence
files only. No tracked historical file, production code, test or workflow changes.
`git diff --cached --check` passes. These checks establish document consistency,
not independent verification of the operator's package or laptop evidence.

Official external URLs were opened in the separate source review; local-link
validation does not assert that a publisher page is immutable or independently
cryptographically authenticated. Hosted CI for the eventual exact PR HEAD is
reported separately in the final delivery; no earlier run is reused as its PASS.
