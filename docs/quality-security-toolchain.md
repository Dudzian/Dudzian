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
- mypy 1.10.0 remains the single type checker. Its configured file scope is
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
  disabled. The redacted baseline contains 20 reviewed public-digest,
  fixture, or example findings, individually justified in
  `docs/security/betterleaks-baseline-review.md`; findings not in that baseline
  block the working-tree gate.
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

This runs the repository Ruff baseline and an offline Betterleaks working-tree
scan. Add `-IncludeTypes` before a PR to run the configured mypy scope. Normal
pre-commit hooks run Ruff, formatting checks, mypy, and the staged secret scan.

### Tier B: normal pull request

The `Quality and Security` workflow runs Ruff, mypy, the property/critical test
set with branch coverage, a 70% diff-coverage ratchet, maintained Semgrep rules
as advisory output, custom Semgrep rules as blocking checks, and the baselined
offline Betterleaks working-tree scan. Existing CI continues to own the larger
pytest, pip-audit, and SBOM jobs.

The production-local issuer-history component uses the existing PostgreSQL 16
service in the serial quality-ratchet job. Its durability and security tests
append branch coverage for the existing history oracle and all three new
PostgreSQL modules to the same coverage report and 70% changed-code gate.
Existing tests and coverage scopes remain enabled. All three new modules enter
the configured mypy scope and its existing Linux and Windows checks; job
timeouts are unchanged.
See [issuer-history limits and qualification](architecture/issuer_history_production_local.md)
for the storage scope and measured test result.

Trwałe przechowywanie Root Proof korzysta z tego samego PostgreSQL 16 i seryjnego
kroku testów historii. Testy `test_postgresql_root_proof_retention.py` dopisują
pokrycie gałęzi obu modułów retention do istniejącego raportu i progu 70%.
Moduły obejmują również istniejące bramki mypy na Linux i Windows oraz sprawdzanie
formatowania. Zakres pozostałych testów, timeouty i bramki sprzętowe pozostają
bez zmian. Granice tego nieautoryzującego magazynu opisuje
[dokumentacja retention](architecture/stage9_durable_root_proof_retention.md).
Drugi, odizolowany i jednorazowy klaster PostgreSQL 16 ma w tym zadaniu testowym
wyłączone `fsync`, aby sprawdzić odmowę instalacji i runtime. Pierwszy klaster
zachowuje trwałe ustawienia. To infrastruktura negatywnego testu CI; nie zmienia
konfiguracji wdrożenia ani bramek kwalifikacji sprzętowej.

Odczytowa kwalifikacja atomowej kompozycji Stage 9 i walidacja RESERVED/STAGED
wchodzą do istniejących bramek Ruff oraz mypy Linux/Windows. Szybkie negatywne
testy kwalifikacji korzystają z seryjnej bramki jakości; testy rzeczywistych
autorytetów i dokładnych bajtów korzystają z semantic issuer preflight oraz
jego niezmienionego progu 70% diff coverage. Drugi trwały PostgreSQL 16 w tym
zadaniu dowodzi odmowy dla różnych klastrów, a restart pierwszego potwierdza
ponowną kwalifikację z katalogu. Niezmienna lineage INITIAL_BINDING i signed
CHA request jest przygotowana raz dla nowego modułu integracyjnego. Timeouty,
dotychczasowe testy i progi pokrycia pozostają bez zmian. Zakres i blokady
opisuje [audyt granicy atomowej](architecture/stage9_qualified_atomic_root_proof_boundary.md).

Trwały signing intent korzysta z dwóch istniejących seryjnych bramek PostgreSQL 16.
Test schematu dopisuje pokrycie gałęzi `postgresql_root_proof_signing_intent_schema.py`
do `coverage.xml` w `quality-ratchet`. Testy runtime z pełną genuine lineage
działają w `semantic-issuer-preflight` i rozszerzają jego
`semantic-preflight-coverage.xml` o `postgresql_root_proof_signing_intent.py`.
Istniejący w tym raporcie `cha_root_proof_signed_attempt.py` obejmuje również
nowy wewnętrzny selector blokady. Obie istniejące bramki changed-code pozostają 70%.
Oba moduły obejmuje skonfigurowany mypy na Linux
i Windows, a moduły i testy obejmuje jawna bramka formatowania Ruff. Istniejące
scope, timeouty i bramki sprzętowe nie są ograniczone.
[Dokumentacja signing intent](architecture/stage9_durable_root_proof_signing_intent.md)
opisuje stan `RESERVED`, izolację `issuer_writer`, exact retry oraz brak
uprawnienia do emisji.

Podział nowych testów zachowuje timeouty 120 i 180 minut oraz wszystkie wcześniejsze
testy i zakresy pokrycia. Komentarz workflow podaje wcześniejsze udane przebiegi
`quality-ratchet` 80–108 minut; 85–100 minut dla istniejącej semantic lineage jest
szacunkiem, a nie pomiarem z tej sesji. Przy szacowanych 60–70 minutach nowego runtime
semantic job wymaga około 145–170 minut. Pełne oba jobs nie zostały tu uruchomione;
lokalna walidacja obejmuje odpowiednie testy PostgreSQL i dwa raporty ze scope
zgodnymi z tym podziałem, obok raportu łącznego.

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

- Signal strength has the signed `[-1, 1]` contract (direction plus magnitude),
  while sizing uses its magnitude. Non-finite or out-of-range strength/confidence
  yields a zero-sized position; calculation errors no longer return a non-zero
  fallback position.
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
