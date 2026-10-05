# Quality, security, and code-intelligence toolchain

This document defines the maintained quality gates for CryptoHunter. Repository
source remains authoritative; Codebase Memory is an index used to narrow source
inspection, not a substitute for reading or testing the implementation.

## Tool choices

- Codebase Memory MCP 0.11.0 is retained for symbol, caller/callee, impact, and
  architecture discovery. Its persistent index lives outside the repository.
- Ruff 0.14.11 remains the single Python linter/formatter. The existing narrow
  rule baseline is retained; broader rules are applied to new files and reviewed
  incrementally rather than rewriting legacy code.
- mypy 1.10.0 remains the single type checker. Its configured 260-file scope is
  a ratchet and must not be silently widened by pre-commit filename arguments.
- pytest remains the test runner. Hypothesis adds property tests only around
  risk, licensing, canonical serialization, and hardware-independent TPM logic.
- coverage.py records branch coverage. diff-cover enforces 70% on changed code;
  this is intentionally a starting ratchet, not a claim that legacy coverage is
  sufficient.
- Semgrep CE 1.179.0 provides an advisory maintained-rules baseline and five
  blocking, tested CryptoHunter rules.
- pip-audit checks the canonical desktop lock file. Dependency updates are
  reviewed and tested; `--fix` is not used automatically.
- Betterleaks 1.9.0 is the only secret scanner. Validation against providers is
  disabled. The redacted baseline contains 28 reviewed fixture/example findings;
  findings not in that baseline block the working-tree gate.
- Dependabot covers pip, the desktop npm project, and GitHub Actions weekly.
- CodeQL runs Python and JavaScript/TypeScript `security-extended` queries.

Bandit, Gitleaks, Renovate, Pyright, ty, Graphify, and SonarQube are not added:
their responsibilities are already covered. `uv` is deferred because the
existing pip/venv workflow is stable. OpenTelemetry is deferred until the runtime
flow and exporter boundary can be designed without coupling trading to a backend.

## Gate tiers

### Tier A: fast local

From PowerShell:

```powershell
./scripts/quality/run_fast.ps1
```

This runs the repository Ruff baseline and an offline Betterleaks scan of changed
files. Add `-IncludeTypes` before a PR to run the configured mypy scope. Normal
pre-commit hooks run Ruff, formatting checks, mypy, and the staged secret scan.

### Tier B: normal pull request

The `Quality and Security` workflow runs Ruff, mypy, the property/critical test
set with branch coverage, a 70% diff-coverage ratchet, maintained Semgrep rules
as advisory output, custom Semgrep rules as blocking checks, and the baselined
offline Betterleaks working-tree scan. Existing CI continues to own the larger
pytest, pip-audit, and SBOM jobs.

### Tier C: heavy and scheduled

CodeQL runs weekly and on normal main/PR events. Full-history Betterleaks runs
weekly or manually as an advisory baseline. A future mutation job should target
only pure functions in licensing verification, risk sizing, and policy-digest
construction on Linux; it should start manual, record runtime and mutation score,
and become scheduled only if the signal justifies its cost.

### Tier D: hardware and qualification

TPM hardware, Stage 9/10 qualification, reboot, destructive lifecycle, and
self-hosted Windows tests retain their existing explicit opt-in gates. The new
workflow never invokes them.

## Security invariants now encoded

- Non-finite or out-of-range signal strength/confidence yields a zero-sized
  position; calculation errors no longer return a non-zero fallback position.
- Non-finite position size, portfolio heat, or sector exposure fails the risk
  gate closed.
- Canonical license JSON round-trips only inside the interoperable integer range,
  rejects non-canonical whitespace, and changes to a signed payload never verify.
- TPM PolicyAuthorize digests remain deterministic and bind key name, policyRef,
  and approved policy without exercising physical TPM hardware.
- Custom Semgrep rules reject request calls without explicit timeouts, literal
  TLS verification disabling, `shell=True`, dynamic eval/exec, and signature
  exception paths that return success.

## Baseline and maintenance notes

- Initial safe test profile: 6,248 passed, 10 skipped, 271 deselected, one failure
  caused by a missing `pytest-asyncio` plugin; that plugin is now present in the
  isolated audit environment.
- Existing Ruff baseline: zero findings, with three legacy warnings for foreign
  `WPS433` noqa codes. Expanded E/F/B/I rules found 2,529 legacy findings, so they
  are not enabled repository-wide.
- mypy baseline and current result: zero errors across 260 files, seven notes.
- Existing selected CI coverage was 81% line and 76% branch. The critical
  property set is intentionally narrower; changed lines currently meet the 70%
  ratchet.
- Maintained Semgrep rules produced 33 advisory findings and six timeouts during
  baseline review. The five custom rules have fixture tests and currently produce
  zero blocking production findings.
- pip-audit found zero actionable vulnerabilities in 98 locked dependencies after
  the repository's documented ignore for disputed `PYSEC-2024-277`.
- Betterleaks originally found a tracked Telegram credential candidate in `.env`
  and `env.example`; both files are sanitized. Because the value exists in Git
  history, its owner must rotate/revoke it and separately decide whether history
  rewriting is justified.

Never lower a gate or extend the Betterleaks baseline merely to make CI green.
Review the finding, add a negative-path regression test where practical, and use
the narrowest documented exception only for a demonstrated false positive.
