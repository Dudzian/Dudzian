# Stage 9 V5d2 preparation — validation record

> Zapis poniżej dotyczy pierwotnego commita
> `dc1c973ecfb0ad35ba91a88081c831e021a3d576`. Wyniki poprawek przeglądu,
> nowe powiązania treści i ograniczenia są w końcowej sekcji
> „Poprawki przeglądu PR #3278”. Historycznych wyników nie należy przenosić
> na późniejsze rewizje ani traktować jako kwalifikacji fizycznego hosta.

This record describes software validation in the managed Linux development
workspace on 2026-11-16. It is not a receipt from the operator's Dell laptop,
an installed-controller preflight, independent approval or physical qualification.

## Revision and content bindings

The clean starting checkout and observed remote `main` are both
`213226f6222f8d1ece05987afe038c82ed61923d`, the merge of PR #3277. Work is on
`codex/stage9-v5d2-operator-preparation`. The draft PR metadata and final report
bind the resulting commit; the following SHA-256 values bind the exact tested
implementation, finalized tests, operator plan and workflow without a
self-referential commit hash in this document.

| File | SHA-256 |
| --- | --- |
| `scripts/stage9_operator_v5d2_inventory.sh` | `bc70213c842e5339996f53f554457186bc040e570fff7abd28c12b71930de2ac` |
| `tests/scripts/test_stage9_operator_v5d2_inventory.py` | `eebf0a9189de20c668cb1b6e429aa4e430d2b6c2c3cdd6d3289511b9cbda5804` |
| `docs/architecture/stage9_operator_v5d2_deployment_plan.md` | `de23d04e0440ca595259d2eb90d302def3bb02b7fd769457ec32345e183c1e00` |
| `.github/workflows/platform-deployment.yml` | `ed3721a47f3462c59080f1590abf637d56f8e286ebe55decb2ff78efd6f27bf2` |

`git diff --exit-code 213226f6222f8d1ece05987afe038c82ed61923d -- deployment docs/architecture/stage9_closure_matrix_v5d1.json`
returns 0: retained controller code and closure-gate statuses are unchanged.

Codebase Memory was unavailable in this session. Contract conclusions were
checked against actual source, documentation and regression tests; no graph
coverage or exhaustive repository-index claim is made.

## Final tests

Working directory for each command is `/workspace/Dudzian`. The isolated
development-tool venv `/workspace/work/stage9-v5d2/venv` is not the protected
observer installation or investigated target interpreter. Tool versions:
pytest 9.2.0, pre-commit 4.7.0, Ruff 0.14.11, mypy 1.10.0, Semgrep 1.179.0,
Betterleaks 1.9.0, cryptography 50.0.2 and defusedxml 0.7.1.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /workspace/work/stage9-v5d2/venv/bin/python -m pytest -q --noconftest tests/scripts/test_stage9_operator_v5d2_inventory.py
```

Exit 0: **39 passed, 0 failed, 0 skipped**, 91.50 seconds. Tests model metadata
and subprocess boundaries, including absent installation/policy/artifacts,
denied access, unsafe ownership/modes, hardlinks, special files, symlinked
ancestors/leaves/launcher, missing tools, unsupported platform, unavailable or
timed-out libvirt, wrong returned URI, revision drift and missing independent
approval. They snapshot retained synthetic state, audit permitted commands,
and confirm candidate Python/QEMU are never invoked. No fake policy file or
approval signature is provisioned on a host.

Hostile inherited logging/authentication settings are checked through real
`env -i`. A separate test uses real Git in a disposable temporary repository
with inherited and repository-configured Trace2 destinations: none creates a
trace file. Fake libvirt verifies only sanitized explicit settings. That test
does not qualify native libvirt authentication backends or a physical socket.
The POSIX shebang harness has an explicit unsupported-platform skip; it did
not skip on this Linux run. Native Git was available and its test ran.

```bash
/workspace/work/stage9-v5d2/venv/bin/python -m pytest -q tests/deployment/test_stage9_observer_collector_identity_v5d1.py tests/deployment/test_stage9_observer_kvm_domain_v5d1.py tests/deployment/test_stage9_observer_kvm_image_v5d1.py tests/deployment/test_stage9_observer_kvm_preflight_v5d1.py tests/deployment/test_stage9_observer_runtime_identity_v5c3.py tests/deployment/test_stage9_observer_pyc_execution_v5c3.py tests/deployment/test_stage9_observer_compile_worker_v5c3.py
```

Exit 0: **659 passed, 0 failed, 1 skipped**, 2.09 seconds. The skip is
`test_native_non_linux_host_refuses_before_policy_or_posix_access`, intentionally
skipped on Linux with reason `Native unsupported-host refusal`. It is not a
pass. Retained test source is unchanged from the baseline.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /workspace/work/stage9-v5d2/venv/bin/python -m pytest -q --confcutdir=tests/scripts tests/scripts/test_platform_core_evidence_workflow.py tests/scripts/test_linux_physical_evidence_workflow.py
```

Exit 0: **137 passed, 0 failed, 0 skipped**, 2.11 seconds. Existing platform
evidence contracts and physical opt-in gates remain intact. The V5d1 static
job now additionally runs V5d2 shell syntax/lint/format/synthetic regressions
and retains its own JUnit artifact; it never runs the live inventory or native
controller in CI. Existing steps, timeout and hardware gates are preserved.

Distinct final test total: **835 passed, 0 failed, 1 skipped**. Earlier
exploratory runs included transient shell-read failures during concurrent
edits and a malformed-revision test expectation correction. Those results
were not counted as passes; the final inventory run used frozen source bytes
bound above. No test or gate was disabled to obtain these results.

## Quality checks

```bash
bash -n scripts/stage9_operator_v5d2_inventory.sh
/workspace/work/stage9-v5d2/venv/bin/python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
PATH=/workspace/work/stage9-v5d2/venv/bin:$PATH PRE_COMMIT_HOME=/workspace/work/stage9-v5d2/pre-commit python -m pre_commit run ruff-format --all-files
/workspace/work/stage9-v5d2/venv/bin/python -m mypy --config-file=pyproject.toml
git diff --check
```

All exit 0. Ruff: `All checks passed!`; the exact all-files formatter hook:
`ruff format...Passed`, with no formatting changes. Configured mypy:
`Success: no issues found in 417 source files` (seven existing annotation
notes). This is the configured Python scope, not a shell type checker.

```bash
SEMGREP_SEND_METRICS=off SEMGREP_SETTINGS_FILE=/workspace/work/stage9-v5d2/semgrep-settings.yml SEMGREP_LOG_FILE=/workspace/work/stage9-v5d2/semgrep-runtime.log /workspace/work/stage9-v5d2/venv/bin/semgrep scan --config .semgrep/cryptohunter.yml --metrics=off --disable-version-check --error bot_core core deployment scripts
SEMGREP_SEND_METRICS=off SEMGREP_SETTINGS_FILE=/workspace/work/stage9-v5d2/semgrep-test-settings.yml SEMGREP_LOG_FILE=/workspace/work/stage9-v5d2/semgrep-test-runtime.log /workspace/work/stage9-v5d2/venv/bin/semgrep scan --test --config .semgrep/cryptohunter.yml .semgrep/cryptohunter.py --metrics=off --disable-version-check --jobs 1
BETTERLEAKS_BIN=/workspace/work/stage9-v5d2/tools/betterleaks /workspace/work/stage9-v5d2/venv/bin/python scripts/quality/betterleaks_hook.py .github/workflows/platform-deployment.yml scripts/stage9_operator_v5d2_inventory.sh tests/scripts/test_stage9_operator_v5d2_inventory.py docs/architecture/stage9_operator_v5d2_deployment_plan.md docs/architecture/stage9_operator_v5d2_validation.md
```

All exit 0. Semgrep: five blocking rules, 1149 Python targets, zero findings;
five of five rule fixtures pass. Its existing large-file and ignore-pattern
exclusions remain (one large file, four matching ignored paths); the custom
Python rules are not shell analysis. Betterleaks: zero leaks on the complete
five-file change, with redaction and provider validation disabled by the
existing wrapper. Its downloaded v1.9.0 binary archive matched the repository
workflow SHA-256 `f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7`.

Initial mypy execution found a missing development-tool `psycopg` dependency;
installing the declared tooling dependency resolved it. Initial Semgrep
startup could not write its default settings in the read-only home; explicit
workspace settings/log destinations resolved it. Neither initial failure is
reported as a pass. No protected installation or system package was changed.
Community advisory Semgrep, full repository pytest, Windows checks, dependency
audit and hosted CI completion are not claimed by this local record.

## Read-only syscall observation and limits

The finalized script was also run on the development cloud host:

```bash
strace -f -qq -e trace=%file,fchmod,fchown,ftruncate -o /workspace/work/stage9-v5d2/inventory-file-syscalls-final.log bash scripts/stage9_operator_v5d2_inventory.sh --repository /workspace/Dudzian > /workspace/work/stage9-v5d2/inventory-cloud-final.stdout 2> /workspace/work/stage9-v5d2/inventory-cloud-final.stderr
```

Exit 0, empty script stderr. Review of 12140 traced file-syscall lines found
47 successful write-mode opens, all to `/dev/null`, zero unexpected successful
write opens and zero successful filesystem mutation calls. Output/trace files
were opened by the external test driver. Failed `/dev/tty` access was not a
successful write. This supplements source review and the synthetic snapshots;
it does not establish all possible native-library behavior on another host.

This cloud host has no available KVM device, QEMU tools or virsh. Protected
path inspection refuses the cloud filesystem's non-root-owned ancestor.
Those results are diagnostics, not evidence of the operator laptop's state.
Live libvirt authentication, installed Python imports, signed policy contents,
domain/image measurements and physical isolation were **NOT_TESTED**. The
operator must separately review the fixed local socket's authentication
boundary before inventory; null auth-file settings do not qualify arbitrary
SASL/GSSAPI backends or authorize production credential access.

## Remaining qualification blockers

The operator-reported laptop still lacks the protected installation, trusted
Python environment and external approval artifacts. A separate independently
reviewed installation/ownership handoff, dependency/native-binding closure,
approved disposable domain, immutable complete image/runtime expectations,
independent policy and trust custody, launch/loaded-code verification,
effective isolation/lifecycle evidence and independent acceptance remain
required. The six existing stopped domains are not selected or altered.
There is no safe retained real-execution admission or self-approval path.

No privileged installation, investigated CPython/bytecode execution, VM
lifecycle operation, image modification, signing-key access, production trust
change or production authorization occurred. The focused change is ready for
independent code review as a draft; deployment and physical qualification
remain blocked.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
HOST_PREFLIGHT_VERIFIED = NOT_CLAIMED
EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
```

## Poprawki przeglądu PR #3278

Walidacja z 2026-11-16 dotyczy poprawki na istniejącej gałęzi
`codex/stage9-v5d2-operator-preparation`, względem przejrzanego commita
`dc1c973ecfb0ad35ba91a88081c831e021a3d576`. Końcowy SHA jest powiązany
z opisem i metadanymi istniejącego PR. Poniższe SHA-256 identyfikują dokładne
zweryfikowane treści; sam hash nie stanowi niezależnej aprobaty źródła.

| Plik | SHA-256 po poprawce |
| --- | --- |
| `scripts/stage9_operator_v5d2_inventory.sh` | `f8916463c89c28f24de63533ba66d05e8d802ac220853bbdc7101d8595abb850` |
| `tests/scripts/test_stage9_operator_v5d2_inventory.py` | `e965a4ccace9ed573c2e68f509ebe19dbdda6cb77bc6f2167a6d56159ad410db` |
| `tests/scripts/test_stage9_safe_ci.py` | `0a8c35d5f8c97560436029f9db3df587ff6dc621665ef7ffb225781af794e922` |
| `docs/architecture/stage9_operator_v5d2_deployment_plan.md` | `c3d3d4e161a72c8d1b7930fc4eea2ad69669020b376c2011eb08a449a3f88b03` |

### Zachowanie i granice

Usunięto zakodowany `EXPECTED_BASELINE`. Każdy poprawny, bieżący SHA odczytany
przez Git daje `repository_revision=PRESENT` z opisem
`OBSERVED_REVISION_NOT_RELEASE_APPROVAL`. Odrębny wynik
`repository_release_approval=UNVERIFIED` z
`INDEPENDENT_SOURCE_REVIEW_REQUIRED` zachowuje obowiązkową niezależną kontrolę
źródła. Inna rewizja, czysty checkout ani historycznie zgodny SHA nie aprobują
wydania, hosta ani wykonania.

Domyślnie skrypt nie kontaktuje libvirt: `libvirt_readonly=NOT_TESTED` oraz
`LOCAL_AUTH_REVIEW_AND_EXPLICIT_OPT_IN_REQUIRED`. Po odrębnym przeglądzie
i zatwierdzeniu lokalnej konfiguracji uwierzytelniania operator może wybrać
`--probe-libvirt-after-local-auth-review`. Flaga wybiera wyłącznie diagnostykę;
nie dowodzi wykonania przeglądu i nie zastępuje żadnej aprobaty bezpieczeństwa.
Natywne backendy mogą korzystać z poświadczeń spoza jawnego środowiska, np.
domyślnego cache Kerberos. Bez wcześniejszej kontroli pozostaje tryb domyślny.
Zachowano stałe odczytowe `qemu:///system`, limit czasu, czyszczenie środowiska,
odmowy i brak eskalacji, alternatywnych URI oraz zmian VM.

Odtworzono też błąd istniejącego testu integracji Actions: asercja pięciu
poleceń nie uwzględniała szóstego kroku V5d2 dodanego w pierwotnym PR.
Poprawiony test zachowuje dokładne asercje pięciu poleceń V5d1 i dodatkowo
sprawdza cztery dozwolone polecenia syntetycznego kroku V5d2, oba artefakty,
SHA/próbę workflow, retencję, uprawnienia i timeout. Workflow nie zmieniono
w tej poprawce; test wymaga, aby CI nie uruchamiało inventory ani sondy natywnej.

### Końcowe testy lokalne

Wszystkie polecenia wykonano w `/workspace/Dudzian`, w tym samym odrębnym
środowisku narzędziowym. Nie użyto chronionego ani badanego interpretera.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /workspace/work/stage9-v5d2/venv/bin/python -m pytest -q --noconftest tests/scripts/test_stage9_operator_v5d2_inventory.py
/workspace/work/stage9-v5d2/venv/bin/python -m pytest -q -rs tests/deployment/test_stage9_observer_collector_identity_v5d1.py tests/deployment/test_stage9_observer_kvm_domain_v5d1.py tests/deployment/test_stage9_observer_kvm_image_v5d1.py tests/deployment/test_stage9_observer_kvm_preflight_v5d1.py tests/deployment/test_stage9_observer_runtime_identity_v5c3.py tests/deployment/test_stage9_observer_pyc_execution_v5c3.py tests/deployment/test_stage9_observer_compile_worker_v5c3.py
/workspace/work/stage9-v5d2/venv/bin/python -m pytest -q --confcutdir=tests/scripts tests/scripts/test_stage9_safe_ci.py tests/scripts/test_history_reproposal_quality_ci.py tests/scripts/test_platform_core_evidence_workflow.py tests/scripts/test_linux_physical_evidence_workflow.py --junitxml=/workspace/work/stage9-v5d2/review-fix-ci-tests.xml
```

| Zakres | Zaliczone | Nieudane | Pominięte | Czas |
| --- | ---: | ---: | ---: | --- |
| V5d2 | 62 | 0 | 0 | 128,78 s |
| V5d1/V5c3 | 659 | 0 | 1 | 2,95 s |
| Kontrakty Actions i dowodów platformy | 215 | 0 | 0 | 7,67 s |
| Łącznie, bez podwójnego liczenia | **936** | **0** | **1** | — |

Pominięto istniejący test
`test_native_non_linux_host_refuses_before_policy_or_posix_access` na Linux,
z powodem `Native unsupported-host refusal`. Nie jest zaliczony. Zachowano
wcześniejsze testy odmów; dodano zmieniające się SHA, rozdzielenie aprobaty,
brak połączeń domyślnie, jawny wybór sondy, kolejność/duplikaty argumentów,
brak zależności przy opt-in i zachowanie niezmiennych statusów.

Przed poprawką kontraktu CI odtworzono `1 failed` (`assert 6 == 5`).
Pierwsza zbiorcza kolekcja testów CI zgłosiła brak narzędziowego `coverage`;
do venv dodano deklarowane `coverage>=7.10,<8` (7.17.3). Po poprawce i
uzupełnieniu narzędzia pełny zestaw dał wynik z tabeli. Wstępnego błędu ani
niezebranych testów nie policzono jako zaliczonych. Nie wyłączono testów ani
bramek.

### Kontrole jakości i bezpieczeństwa

```bash
bash -n scripts/stage9_operator_v5d2_inventory.sh
/workspace/work/stage9-v5d2/venv/bin/python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
/workspace/work/stage9-v5d2/venv/bin/python -m mypy --config-file=pyproject.toml
/workspace/work/stage9-v5d2/venv/bin/python -m mypy --config-file=pyproject.toml --platform win32 --cache-dir=/workspace/work/stage9-v5d2/mypy-win32
PATH=/workspace/work/stage9-v5d2/venv/bin:$PATH PRE_COMMIT_HOME=/workspace/work/stage9-v5d2/pre-commit BETTERLEAKS_BIN=/workspace/work/stage9-v5d2/tools/betterleaks python -m pre_commit run --all-files
PATH=/workspace/work/stage9-v5d2/venv/bin:$PATH PRE_COMMIT_HOME=/workspace/work/stage9-v5d2/pre-commit python -m pre_commit run ruff-format --all-files
git diff --check
git diff --cached --check
```

Wyniki: PASS; mypy Linux i Win32: po 417 plików, zero błędów. Pełny pre-commit
zaliczył Ruff, formatowanie, mypy i Betterleaks. Obowiązkowy osobny hook
`ruff-format --all-files` również zaliczony bez zmian formatowania.
Powtórzono dokładne polecenia blokujących reguł Semgrep i ich fixtures
z wcześniejszej sekcji: pięć reguł, 1149 celów, zero znalezisk oraz 5/5
fixtures PASS. Zachowano istniejące wyłączenia dużych/ignorowanych plików;
reguły Python nie są analizą semantyki powłoki. Skan Betterleaks obejmuje
całą bieżącą pięcioplikową poprawkę, bez nowych wyjątków.

```bash
/workspace/work/stage9-v5d2/venv/bin/python -m pip_audit --cache-dir /workspace/work/stage9-v5d2/pip-audit-cache --requirement deploy/packaging/requirements-desktop.lock --format json --output /workspace/work/stage9-v5d2/review-fix-pip-audit-report.json
```

Kod 0: 104 zależności, zero znanych podatności, zero pominiętych zależności;
bez `--ignore-vuln` i bez `--fix`. Początkowa próba nie mogła utworzyć cache
w katalogu domowym tylko do odczytu. Jawny katalog cache w workspace rozwiązał
problem środowiska; nie zmieniano locka ani zakresu audytu.

### Odczytowa obserwacja syscall i ograniczenia

```bash
strace -f -qq -e trace=%file,fchmod,fchown,ftruncate -o /workspace/work/stage9-v5d2/review-fix-inventory-file-syscalls.log bash scripts/stage9_operator_v5d2_inventory.sh --repository /workspace/Dudzian > /workspace/work/stage9-v5d2/review-fix-inventory-cloud.stdout 2> /workspace/work/stage9-v5d2/review-fix-inventory-cloud.stderr
```

Kod 0, stderr skryptu pusty. W 12124 wierszach zaobserwowano 46 udanych
otwarć z flagami zapisu, wszystkie do `/dev/null`; zero innych udanych
otwarć do zapisu, zero udanych mutacji systemu plików i zero uruchomień
`virsh`. Bieżący `dc1c973ecfb0ad35ba91a88081c831e021a3d576` odczytano jako
obserwację `PRESENT`, z odrębną aprobatą `UNVERIFIED`; sonda pozostała
`NOT_TESTED`. To obserwacja środowiska deweloperskiego, nie laptopa operatora
i nie kwalifikacja hosta. Testy jawnej sondy używały atrap, nie natywnego
połączenia libvirt.

Wyników GitHub Actions nie wyprowadza się z testów lokalnych ani starego SHA.
Opis PR i raport końcowy podają odczytany status dla nowego SHA; workflow
w toku, w kolejce lub pominięty nie jest zaliczonym workflow.
Nie instalowano kontrolera, nie zmieniano VM/polityki operatora/macierzowych
bramek i nie wykonywano badanego CPython. Pozostają niezależna kwalifikacja
hosta, chroniona instalacja i zależności, aprobaty/trust custody, osobny
disposable domain, obrazy i runtime, pomiary izolacji i niezależny launch.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
HOST_PREFLIGHT_VERIFIED = NOT_CLAIMED
EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
```
