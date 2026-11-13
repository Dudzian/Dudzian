# Observer 3A reconciliation — validation, 2026-11-13

Source baseline: `d011240f3f5a81b41d1f81ef179efd763c641b30`.
Change: three new documentation/evidence files only. No historical file,
implementation, test, security policy or production status is changed.
See the [review and requirement table](../stage9_observer_3a_reconciliation_2026-11-13.md)
and [operator ledger](stage9_observer_3a_operator_reconciliation_2026-11-13.json).

## Actual Cloud regression runs

Existing `.venv`: Python **3.11.17**, pytest **9.2.0**. No dependency was
installed. These development tests ran in Cloud; they do not qualify the
physical Ubuntu 24.04 host or its exact CPython 3.12 runtime.

| Command scope | Measured result |
| --- | --- |
| Contract, offline, Zstandard, TAR V3 and package V3 suites | 1332 passed in 6.46 s; exit 0 |
| Real-DEB V2/V3 reproduction-tool suites, synthetic fixtures only | 55 passed in 0.51 s; exit 0 |

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=work/observer3a-review/pytest tests/test_stage9_observer_contract.py tests/deployment/test_stage9_observer_offline.py tests/deployment/test_stage9_observer_zstd.py tests/deployment/test_stage9_observer_tar_v3.py tests/deployment/test_stage9_observer_package_v3.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=work/observer3a-review/pytest-real-deb tests/deployment/test_stage9_real_deb_v2.py tests/deployment/test_stage9_real_deb_v3.py
```

Temporary fixture files were subsequently moved outside the checkout to
`/workspace/work/observer3a-review/fixtures`. The synthetic DEBs created by
tests are not the absent original operator materials. No real DEB comparison,
host scan, target CPython/import, installation, maintainer script, qualification
or production action ran.

## Style and security gate limitations

[AGENTS.md](../../../AGENTS.md) requires the exact command
`python -m pre_commit run ruff-format --all-files`, including documentation-only
PRs. The command was attempted and **failed**, exit 1: the default Cloud
Python 3.12.14 has no `pre_commit` module. The existing `.venv` has pre-commit,
but no provisioned hook environment was found in the inspected default,
workspace or temporary caches. Initializing it would require installation;
the user's explicit no-install instruction takes precedence. This gate is
**not passed**; the PR must remain **draft** pending the actual style workflow
and required review. No hook configuration or scope was changed.

As an additional diagnostic, existing Ruff **0.14.11** checked all **2516
tracked Python files matching the configured ruff-format hook's include and
exclude patterns**, with `format --check --no-cache`. Exit **1**: 2512 already
formatted, four pre-existing files would be reformatted:

- `bot_core/generated/__init__.py`
- `bot_core/generated/trading_pb2.py`
- `bot_core/generated/trading_pb2_grpc.py`
- `bot_core/runtime/archive/stage6_hypercare.py`

None is changed by this documentation-only review. The diagnostic is not a
replacement for the required pre-commit gate, and does not justify rewriting
unrelated historical/generated code in this PR.

The Tier A Ruff lint scope plus the five Observer implementation modules
passed, exit 0:

```sh
.venv/bin/python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui deployment/stage9_observer_contract.py deployment/stage9_observer_offline.py deployment/stage9_observer_ubuntu.py deployment/stage9_observer_tar_v3.py deployment/stage9_observer_package_v3.py
```

The existing Betterleaks wrapper was attempted on the two initial new report
files and returned **exit 2**, reporting that the required executable was
unavailable. No scan result is claimed, including for this subsequently added
validation file. No scanner was installed, substituted or suppressed. No new
coverage, mypy, Semgrep, vulnerability-scan or remote CI result is claimed by
this packet; earlier reports' results remain historical.

Local document checks verified JSON syntax without duplicate keys, all seven
operator-origin labels, unchanged blocked disposition, the two historical
digest references and all 24 local Markdown links. `git diff --cached --check`
passed. The staged change contains only the three new files; historical
evidence and source files are unchanged.

Source inspection and the independent parallel review found no new confirmed
P0/P1 defect in the inspected paths. Codebase Memory was unavailable. This is
a bounded review, not a repository-wide proof. The new JSON records **operator
claims**, not machine-accepted evidence or an authority contract.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```
