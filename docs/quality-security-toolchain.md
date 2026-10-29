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

The bounded Stage 9 offline artifact verifier enters these existing Ruff,
Linux/Windows mypy, pytest and branch-coverage gates. Its tests perform real
unprivileged file reads and inspect synthetic DEB fixtures as data. These checks
do not qualify a physical installation, runtime execution or host PID scope;
independent acceptance and production qualification remain blocked. Existing
coverage thresholds, job timeouts and hardware opt-in gates remain unchanged.

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

Atomic Root Proof Commit Authority V1 ma osobny, seryjny job PostgreSQL 16
`atomic-root-proof-commit`, z limitem 120 minut. Istniejące joby, ich timeouty
i zakresy testów pozostają bez zmian. Nowy job używa dwóch odizolowanych
klastrów oraz współdzielonej niezmiennej lineage INITIAL_BINDING. Sprawdza
fizyczny write set, ACL, migrację, rollback, lifecycle locks, utratę odpowiedzi
COMMIT i restart. Pięć nowych modułów obejmuje branch coverage oraz niezmieniony
próg diff coverage 70%; wchodzą również do konfiguracji mypy Linux/Windows
i jawnej bramki formatowania Ruff. Zmienne infrastruktury testowej wybierają
klastry i kontroler restartu; nie odblokowują publicznego issuera.
Granice kwalifikacji opisuje
[Atomic Commit V1](architecture/stage9_atomic_root_proof_commit_authority_v1.md).

Durable Custody Lifecycle Fencing V2 rozszerza ten sam seryjny job
`atomic-root-proof-commit` o rzeczywiste PostgreSQL 16, lokalne custody,
administracyjne close/completion, recovery po utracie odpowiedzi COMMIT oraz
wyścig niezależnych procesów issuer/admin. Zachowuje wszystkie testy Atomic
Commit V1, limit 120 minut i ten sam raport `atomic-root-proof-coverage.xml`
z progiem diff coverage 70%. Pokrycie gałęzi obejmuje
`local_signing_lifecycle_evidence.py`, `postgresql_custody_lifecycle.py`,
`postgresql_custody_lifecycle_schema.py` i
`postgresql_custody_lifecycle_qualification.py`. Wszystkie cztery moduły wchodzą
do skonfigurowanego mypy Linux/Win32. Moduły, nowe testy bezpieczeństwa i kontrakt
architektoniczny obejmuje jawna bramka formatowania Ruff. Nie dodaje to bramki
kwalifikacji sprzętowej ani prawa do produkcyjnej emisji Root Proof.
Granice izolacji i trwałe punkty protokołu opisuje
[Lifecycle V2](architecture/stage9_durable_custody_lifecycle_v2.md).

S9-RPI-01 rozszerza ten sam job o niezależny, zamknięty production lifecycle
inventory, jego katalog SQL, procesową granicę instalacji i wspólny guard
właściciela transakcji. Nowe testy używają odrębnych baz PostgreSQL 16,
rzeczywistych procesów, restartu i utraty odpowiedzi COMMIT; nie tworzą
produkcyjnych kluczy ani BIND. Lokalna kwalifikacja SQL z testowym `trust`
nie dowodzi produkcyjnego peer authentication: zainstalowana granica osobno
wymaga dokładnego PostgreSQL `system_user=peer:<unix_user>`.
Nowe moduły obejmuje istniejący scope mypy Linux/Windows, format gate,
raport branch coverage i niezmieniony próg changed-code 70%.
Limity jobów, wcześniejsze testy i bramki sprzętowe pozostają bez zmian.
Zakres i blokady opisuje
[S9-RPI-01](architecture/stage9_production_root_proof_lifecycle_isolation.md).

S9-RPI-02 adds the independent signer IPC, process-owned custody and signer
service modules to the same Linux/Windows mypy scope and explicit Ruff format
gate. The five signer test modules join the existing serial
`atomic-root-proof-commit` job; all three implementation modules enter both raw
coverage reports combined by `atomic-root-proof-coverage`. Existing tests,
timeouts, hardware gates and the 70% changed-code threshold remain in place.
The isolated test environment exercises real local processes and PostgreSQL 16;
it does not qualify a production installation or enable Root Proof issuance.

For a targeted local run, configure the two isolated durable PostgreSQL 16 DSNs
and restart/crash controllers described by the existing atomic job, then run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0 \
python -m pytest -p asyncio -p pytest_cov.plugin \
  tests/security/test_production_signer_ipc.py \
  tests/security/test_production_signer_custody.py \
  tests/security/test_production_signer_deployment.py \
  tests/security/test_production_signer_lifecycle.py \
  tests/security/test_production_signer_boundary.py \
  -m "not hardware and not destructive and not network" \
  --cov=bot_core.production_signer_ipc \
  --cov=bot_core.production_signer_custody \
  --cov=deployment.production_signer_service \
  --cov-branch --cov-config=.coveragerc --cov-report=term-missing -q
```

This targeted loop supplements the unchanged existing atomic qualification and
combined coverage gate. Before a PR, run the mandatory
`python -m pre_commit run ruff-format --all-files`, explicitly check formatting
of `deployment/production_signer_service.py`, and run configured `python -m mypy`
and `python -m mypy --platform win32` without filename arguments. Existing
blocking custom Semgrep rules and the baselined offline Betterleaks scan cover
the new modules through their current directory scopes.

The isolated signer HWID change adds `security/custody_hwid.py` and
`security/keyring_storage.py` to configured Linux/Windows mypy and the explicit
Stage 9 Ruff format list, together with the new custody HWID tests and process
helper. Ordinary `py-tests-ubuntu` collection includes the provider and
keyring unit tests; the existing `integration-linux` job selects the process
tests through their `integration` marker and executes the helper in disposable
children. No new production/hardware job is introduced. The existing atomic
and history test inventories, coverage scopes, ratchets, timeouts and manual
Stage 10 gates remain unchanged. Focused branch coverage and exact executed
commands belong in the PR; synthetic hardware and native-backend fixtures do
not qualify an installed systemd credential mount or production custody.

Adaptery przygotowania RESERVED/STAGED dla inventory V2 rozszerzają ten sam
seryjny job `atomic-root-proof-commit` o testy nowego przygotowania po migracji,
kwalifikacji preparation ACL, signer enrollment, exact retry i commit właśnie
tego nowego kandydata. Trzy moduły V2 obejmują skonfigurowany mypy Linux/Windows,
formatowanie Ruff i branch coverage. Raport obejmuje również współdzielone
czytniki oraz granice autoryzacji zmienione przez ten PR. Pozostają wszystkie
wcześniejsze scenariusze, timeout 120 minut oraz próg diff coverage 70%.
Prywatne ponowne użycie świeżo zweryfikowanego snapshotu w preflight
`cha_root_proof_signed_attempt.py` jest również objęte tym raportem. Moduł
był już w skonfigurowanym mypy, format gate i wcześniejszych coverage scopes.
Nie aktywuje to produkcyjnego issuera ani kwalifikacji sprzętowej.
Zakres opisuje [przygotowanie V2](architecture/stage9_postgresql_v2_root_proof_preparation.md).

History Successor Reproposal V1 ma osobny seryjny job
`history-successor-reproposal`, z własną parą trwałych PostgreSQL 16 i limitem
120 minut. Obejmuje append-only candidate lineage, autorytatywne supersession,
genuine stale-predecessor recovery i rzeczywiste fault/restart tests. Importowane
negatywne przypadki zbiera główny moduł testów, bez osobnego uruchamiania helpera.
Oryginalny `atomic-root-proof-commit` zachowuje wszystkie dotychczasowe testy,
zakresy pomiaru i limit 120 minut. Job `atomic-root-proof-coverage` wymaga sukcesu
obu kwalifikacji i ich surowych danych branch coverage. Łączy je przez
`coverage combine`, następnie generuje XML i egzekwuje niezmieniony próg
`diff-cover --fail-under=70`. Brak wymaganych danych powoduje failure. Pomiary
nowej kwalifikacji obejmują cztery nowe i wszystkie współdzielone moduły atomic,
także nowy authenticated snapshot w `postgresql_root_proof_preparation_v2`.
Dotychczasowe joby `quality-ratchet`, `quality-ratchet-postgresql` oraz
`quality-ratchet-coverage` pozostają niezależne i zachowują swój ratchet.
W ogólnym CI macierz PostgreSQL ma trzy rozłączne shardy: `issuer-durability`,
`history-reproposal` i `remaining-security`. Każdy ma własne klastry; wspólny
zakres markerów i ignorowanych plików pozostaje zachowany. `integration-linux`
wymaga sukcesu całej macierzy. Pure evidence i architecture tests pozostają
w ogólnym shardzie bez `external_postgresql`. Cztery nowe moduły obejmują
również skonfigurowany mypy Linux/Windows i pełny Ruff format preflight.
Odrębny test import graph chroni publiczne read-only porty przed dostępem do
administracji reproposal. Granice i immutable evidence opisuje
[protokół V1](architecture/stage9_history_successor_reproposal_v1.md).

S9-RPI-05 dodaje `independent_custody_completion_qualification.py` do
skonfigurowanego mypy Linux/Windows i obu surowych zakresów branch coverage
łączonych przez istniejący atomic gate. Nowy test qualification-only działa
w tym samym seryjnym jobie; moduł i test obejmuje jawny Ruff format gate.
Pozostają dotychczasowe testy, timeouty, próg 70% i osobne bramki sprzętowe.
Format claims i realny odczyt journalu nie kwalifikują źródła mutation,
CCA ani produkcyjnego completion; granice i fizyczną procedurę opisuje
[S9-RPI-05](architecture/stage9_independent_custody_completion_authority.md).

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

The `test_production_signer_linux_processes.py` harness requires root inside a
disposable Linux container (default capabilities; no privileged container). It
forks two signer identities and an issuer, tests actual Unix DAC and peer
credentials, and uses only synthetic fixture files. Ordinary non-root CI skips
it; this is separate from installed systemd/native-keyring qualification.

S9-RPI-06 adds the read-only `authenticated_custody_mutation_source.py`
protocol verifier to configured mypy Linux/Win32, explicit Ruff formatting and
both raw coverage inventories of the existing atomic/history qualification.
Its security tests and the separate controlled TEST source process harness run
in the serial `atomic-root-proof-commit` job. The exact CI inventory contract
requires these additions while retaining all earlier tests, the existing job
timeouts and the 70% changed-code coverage threshold. The TEST harness has no
production admission, provisioning or completion capability; matching a fixture
Ed25519 signature does not qualify execution, durability or an installed trust
anchor. Real PostgreSQL 16 and the existing root Linux DAC/peer harness remain
separate regression evidence. See
[the S9-RPI-06 protocol and qualification limits](architecture/stage9_authenticated_custody_mutation_source.md).

S9-RPI-07 dodaje `independent_cca_decision.py` oraz jednoznacznie testowy
`test_only_cca_decision_log.py` do skonfigurowanego mypy Linux/Win32, jawnego
Ruff format gate i obu surowych zakresów branch coverage łączonych przez
`atomic-root-proof-coverage`. Testy `test_independent_cca_decision.py`,
`test_cca_durable_decision_log.py` oraz
`test_stage9_independent_cca_boundary.py` rozszerzają istniejący seryjny job
`atomic-root-proof-commit`. Exact inventory sprawdza pełne dotychczasowe
zbiory testów i modułów obu jobów, brak duplikatów, configured mypy i nowe
ścieżki formatowania. Zachowane są timeouty 120/120/25 minut, próg changed-code
70%, marker exclusions oraz osobne bramki sprzętowe. Testowy filesystem log
na Linux i poprawny podpis syntetycznej decyzji nie kwalifikują produkcyjnej
CCA, jej instalacji, committed-close readera ani niezależnego rollback domain.
Granice i fazy recovery opisuje
[S9-RPI-07](architecture/stage9_independent_cca_durable_decision_log.md).

Stage 41B adds a separate synthetic physical signer qualification entrypoint,
`scripts/qualify_stage41b_signer_isolation.py`, and test-only role worker
`deployment/stage41b_role_worker.py`. Both enter the configured Linux/Windows
mypy scope; the worker enters explicit Ruff lint/format gates. Unprivileged
`tests/security/test_stage41b_signer_isolation.py` regressions enter the existing
serial Stage 9 gate and append branch coverage for both new modules without
changing its 70% diff-coverage threshold. Those regressions validate harness
logic and refusal paths only. Physical `--execute` is never ordinary pytest or
PR CI; `--check` cannot award `STAGE41B_TEST_ONLY_PASS`. See the
[physical runbook and evidence limits](architecture/stage9_stage41b_physical_signer_isolation.md).

Stage 41B preparation now has a separate standard-library installer and immutable
PyPI wheel lock for CPython 3.12 on Ubuntu 24.04 x86_64. The unprivileged builder
exports Git object proofs; the independently verified privileged bootstrap checks
those proofs and wheel hashes before offline installation. Prepared-runtime
content and ownership checks are mandatory for physical qualification. Preparation
never starts the physical test. The added preparation module enters the same
configured mypy and coverage scopes, with unprivileged regressions. The explicit
atomic CI inventory contract includes the three Stage 41B test/module entries
while preserving all original atomic and history inventories and the existing
70% coverage threshold. See the
[operator procedure](architecture/stage9_stage41b_operator_runbook.md).

Kontrolowany operator fizycznej instalacji Source/CCA rozszerza ten sam seryjny
job o testy pakietu TEST ONLY, pochodzenia zasobów, odczytowych sond hosta, odmów, awarii, cleanup
i CLI. Trzy nowe moduły deployment oraz interfejs operatorski wchodzą do
skonfigurowanego mypy Linux/Win32, Ruff lint/format i obu surowych zakresów
branch coverage atomic/history. Kontrakt inventory zachowuje wcześniejsze
testy, timeouty, markery i próg 70%. CI wykonuje wyłącznie testy bez uprawnień
administracyjnych; nie wywołuje fizycznego `--execute`. Pełne `MATCHED` bramki
nadal zwraca kod CLI 2 i nie kwalifikuje produkcji. Zobacz
[procedurę Source/CCA TEST ONLY](architecture/stage9_source_cca_physical_test_operator.md).

Naprawa rootless procfs rozdziela obserwację widocznych UID/GID/grup i cgroup
od pełnego skanowania executable jako root. Regresje
`tests/security/test_stage9_source_cca_test_rootless.py` wchodzą do istniejącego
seryjnego inventory i jawnego format gate. Kontrole nieczytelnych lub zmiennych
metadanych pozostają fail-closed; zwykła inspekcja nie przyznaje pełnego CLEAN.
Osobny `--inspect-cleanup-admin` jest jawny i wyłącznie odczytowy. Fizyczny
`--execute` pozostaje poza CI, a status `PHYSICAL_TEST_PENDING` i produkcyjne
blokady pozostają bez zmian. Progi coverage i dotychczasowe testy Stage 41B
nie są osłabiane.

Kontrola pochodzenia kodu operatora Source/CCA wymaga niezależnie przygotowanego
i zewnętrznie zweryfikowanego checkoutu oraz chronionego interpretera i środowiska
importów przed uruchomieniem Pythona jako root. Import wykonuje kod wcześniej
niż kontrola po imporcie; lokalne hashe i obserwacja już załadowanych modułów
nie ustanawiają zaufania do własnego procesu. Dotyczy to także istniejącego
bytecode, plików `.pth` i mechanizmów importu; samo `-B` nie wyłącza ich odczytu.
Regresje weryfikacji kodu obejmują niechronione pliki, zmianę uprawnień bez
zmiany treści, moduły wykorzystywane później oraz odmowę przed zmianami hosta.
Nie uruchamiają fizycznego `--execute` ani sudo w CI. Granice tej lokalnej
kontroli opisuje [procedura operatora](architecture/stage9_source_cca_physical_test_operator.md);
`PHYSICAL_TEST_PENDING` i `PRODUCTION_NO_GO` pozostają bez zmian.
