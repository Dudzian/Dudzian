# Windows self-hosted CI runbook

This workflow moves Windows runs off GitHub-hosted runners and onto the self-hosted Windows machine.

## Runner requirements

- Registered self-hosted runner with labels: `self-hosted`, `Windows`
- Python 3.11 available in `py` launcher or PATH
- `git` and `PowerShell` available
- Optional: pre-populated wheelhouse directory if offline

## How to trigger

1. Open **Actions → Windows Self-Hosted CI → Run workflow**.
2. Inputs:
   - **jobs**: comma-separated set of jobs to run (e.g. `prepare-wheelhouse,ui-packaging-windows`).
   - **wheelhouse_source**: `artifact` (download `wheelhouse-windows-py3.11`), `local_path` (use the provided path), or `none`.
   - **wheelhouse_local_path**: path on the runner when using `local_path`.
   - **pyside6_version**: override PySide6 version if needed (default `6.7.0`).
   - **upload_artifacts**: keep enabled to collect `artifacts/local-ci`, `test-results`, and `wheelhouse`.

## Wheelhouse delivery (offline)

- Preferred: re-use `prepare-wheelhouse` job in this workflow with `wheelhouse_source=artifact` and `upload_artifacts=true`.
- Air-gapped: copy a zipped `wheelhouse` directory to the runner and supply `wheelhouse_source=local_path` pointing to it.
- Verify content before running jobs:
  ```pwsh
  Get-ChildItem -Recurse <wheelhouse>
  ```

## Debugging runner pickup

- Ensure the runner shows as **Idle** in repository settings and has labels `self-hosted` and `Windows`.
- Check the workflow uses `runs-on: [self-hosted, Windows]` (already configured).
- If the job stays queued, confirm the runner service is running and has permission to the repo.

## Typical commands executed

All jobs delegate to `scripts/ci/run_windows.ps1` (wheelhouse-aware):
- `-Job prepare-wheelhouse`: builds wheelhouse offline cache (PySide6 + project deps).
- `-Job ui-packaging-windows`: runs Qt bundling.
- `-Job lint-and-test`, `-Job bot-core-fast-tests`, `-Job ui-native-tests`, `-Job release-quality-gates`: mirror CI commands, using wheelhouse when provided.

Artifacts are written under `artifacts/local-ci/<job>/<timestamp>` and uploaded when `upload_artifacts=true`.
