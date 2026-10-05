# CryptoHunter Codex instructions

## Codebase Memory usage policy

For repository-wide, unfamiliar-code, cross-module, architectural, dependency,
call-path, regression-risk, or impact-analysis tasks:

1. Query the `codebase-memory` MCP server first to identify relevant symbols,
   modules, files, callers, callees, and dependency paths. Prefer
   `search_graph`, `get_architecture`, `trace_path`, `query_graph`,
   `detect_changes`, `get_file_outline`, and `search_code` as appropriate.
2. Use structural memory to narrow the investigation to roughly 3-10 likely
   files or symbols, then open and inspect the actual source files.
3. Treat repository source code as the source of truth. Never make an
   implementation decision from graph output alone.
4. Verify every security-critical, TPM, licensing, deployment, Windows,
   Stage 9, Stage 10, qualification, and CI conclusion directly against source.
5. Before making an exhaustive or negative claim, page through all relevant
   graph results and use `check_index_coverage` for the cited paths or scope.
   A clean coverage result means no recorded gap, not proof of completeness.
6. Prefer targeted source reads after graph discovery over reading broad parts
   of the repository. Expand the source search when graph results are missing,
   ambiguous, stale, or contradicted by the implementation.
7. Skip Codebase Memory for a small task when the exact file and symbol are
   already known and a graph query would add unnecessary overhead.
8. Refresh the index after significant edits, branch changes, rebases, or when
   freshness is uncertain. Use:

   `codebase-memory-mcp cli index_repository --repo-path "C:\Users\kamil\Documents\GitHub\Dudzian"`

9. After editing code, validate with source inspection and the relevant tests;
   the graph does not establish correctness.
10. PowerShell, YAML, JSON, TOML, Markdown, shell scripts, and generated or
    ignored paths may have weaker structural coverage than Python. Use ordinary
    repository search and direct reads to close those gaps.

Target workflow for large tasks:

`query structural memory -> identify 3-10 likely files/symbols -> inspect actual source -> edit -> test`

Do not optimize context usage at the expense of correctness. In TPM, security,
licensing, and deployment analysis, uncertainty requires direct source checks.

## AI-generated code quality policy

1. Treat repository source as the source of truth; Codebase Memory narrows scope
   but never proves behavior or security.
2. After implementation, run the smallest relevant quality gate and tests. New
   behavior requires tests; bug fixes should include a regression test when
   practical.
3. Security-, licensing-, TPM-, exchange-, and risk-critical changes require a
   negative-path test. Failures, malformed input, timeout, `None`, NaN, and
   infinity must fail closed where they cross a security or order boundary.
4. Do not bypass the central risk gate, signature verification, licensing
   verification, or trusted serialization abstractions.
5. Do not disable tests, Ruff, mypy, Semgrep, secret scanning, or coverage merely
   to make CI pass. Do not add `noqa`, `type: ignore`, scanner ignores, or lower a
   coverage threshold without the narrowest documented justification.
6. Preserve the manual/self-hosted safety gates around Stage 10, destructive,
   reboot, and physical-hardware qualification. Never run them as ordinary PR CI.
7. Resolve false positives with the narrowest rule/path exception and record why;
   never suppress an entire security class for one finding.
8. Before finishing, inspect the diff for unrelated formatting or generated-file
   churn and run the relevant Tier A or Tier B commands documented in
   `docs/quality-security-toolchain.md`.
