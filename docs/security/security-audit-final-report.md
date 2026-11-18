# Audyt bezpieczeństwa CryptoHunter — raport stanu

Data: 2026-11-18. Issue: [#3280](https://github.com/Dudzian/Dudzian/issues/3280).
Aktualny zweryfikowany `main`: `3462968720e858f4a8557454b5342375840f1f71`.

**Nie rekomenduje się zamknięcia audytu ani wdrożenia.** Ten dokument zamyka
raportowanie wykonanej transzy; wskazuje także niezakończone kontrole.
`CONFIRMED_FIXED` poniżej dotyczy jawnie wskazanej gałęzi draft PR.
Poprawki nie zostały scalone do `main` i nie są dowodem naprawy instalacji.

## 1. Podsumowanie zarządcze

Potwierdzono scalenie #3279 (cloud plaintext guard), #3282 (Stage 9 V5d4)
i późniejszego #3284. Nie odnaleziono innych zdalnych poprawek poprzedniego
audytu. Poprzednie środowisko i niezapisane dowody oznaczono `UNAVAILABLE`.
Nie wykorzystano dawnych liczb testów jako wyników obecnej sesji.

Odtworzono i zweryfikowano siedem logicznie odrębnych zmian: kompletność
Semgrep, parsowanie XML, świeżość auth cloud, skanowanie indeksu Git,
odzyskiwanie importu offline, zależności npm/renderer i granicę nasłuchu
pozostałych usług gRPC. Wszystkie zapisano jako draft PR, bez merge.

Otwarte blokady obejmują niekompletne rzeczywiste skany Semgrep, nierozstrzygnięte
kandydaty w historii Git, podatną zależność narzędziową npm, oczekujące Actions
oraz brak natywnej i produkcyjnej kwalifikacji. Nie wykazano bezpieczeństwa
całego procesu aktualizacji, wszystkich granic uprawnień ani całej bazy danych.

Codebase Memory nie było dostępne. Ustalenia oparto na źródłach, rzeczywistych
wywołaniach, testach negatywnych i aktualnym GitHub. Brak sygnałów skanera
nie był kryterium uznania komponentu za bezpieczny.

## 2. Powierzchnia ataku i model zagrożeń

| Komponent / wejście | Potwierdzona ścieżka i granica | Zakres oceny |
| --- | --- | --- |
| Qt / QML / PySide | `ui/src/main.cpp`, `ui/src/app/Application.cpp`, `python -m ui.pyside_app` | Kod uruchomienia i wybrane kontrolery; brak natywnej kwalifikacji |
| Lokalny runtime | `scripts/run_local_bot.py` → `bot_core/api/server.py` | Lokalny gRPC; nowy guard przed alokacją; auth lokalne wymaga odrębnej oceny |
| Gateway desktop | `bot_core/runtime/local_gateway.py` | JSON przez stdin/stdout, walidacja Pydantic i token bucket; bez HTTP cookies |
| Cloud | `scripts/run_cloud_service.py` → `CloudRuntimeService` → `LocalRuntimeServer` | Scalony loopback guard; nowy dwukrokowy challenge i sesje w #3288 |
| Risk / metrics | Bootstrap i aktywne launchery → `RiskServer` / `MetricsServer` | Tokeny/scopes i gałęzie TLS; #3290 ogranicza plaintext |
| Market data | Eksportowany `MarketDataServer` | Konstruktor sprawdzony; nie potwierdzono osobnego aktywnego launchera |
| HTTP | Fabryka `bot_core/services/risk_dashboard.py:create_app` | Odczytowe endpointy i limit listy; nie znaleziono aktywnego launchera produkcyjnego |
| Electron / renderer | `ui/desktop/src/main/main.ts` i aplikacja React | Aktualizacja zależności, build, formularze i Chromium E2E; native `NOT_TESTED` |
| Sekrety giełd | `core/security/secret_store.py`, `KeyringSecretStorage` | Odczyt kodu i testy z syntetycznym backendem; bez kwalifikacji systemowego keychaina |
| Dane i autorytety | SQLite persistence, SQLAlchemy manager, moduły PostgreSQL | Przegląd wybranych parametrów, ról/ACL i transakcji; testy kontraktów bez live PG |
| Aktualizacje | QML → Python importer; osobny CLI i osobny instalator C++ | Naprawiono obsłużone błędy importera; end-to-end poza kwalifikacją |
| CI i raporty | Git index, Betterleaks, Semgrep, parsery coverage/JUnit, CodeQL/SBOM | Negatywne testy odmów, skany i odrębne statusy kompletności |

Model uwzględnia niezaufane dane klienta, dokumenty i artefakty, lokalny proces
bez uprawnień operatora, błędną konfigurację listenera, rozbieżność indeksu Git
z working tree, zależności oraz awarie operacji zapisu. Loopback ogranicza
ekspozycję sieciową, ale nie uwierzytelnia procesów na tym samym hoście.
Nie wykonywano handlu live, instalacji uprzywilejowanej, reboot ani prób
Stage 3B, fizycznego TPM lub fizycznej kwalifikacji Linux/Windows.

`TradingStubServer`, dane demonstracyjne i archiwa nie były traktowane jako
dowód bezpieczeństwa produkcyjnego backendu. Sygnały z narzędzi test-support
oceniano według ich rzeczywistych uprawnień i źródeł danych.

## 3. Rejestr ustaleń

Poziomy są oceną wpływu × ekspozycji × prawdopodobieństwa w przejrzanej ścieżce;
nie są stwierdzeniem skutecznego ataku na produkcję.

| ID | Severity / exposure / likelihood | Status | Lokalizacja i dowód | PR / regresja | Pozostałe ryzyko |
| --- | --- | --- | --- | --- | --- |
| CH-SEC-01 | High / CI / medium | `CONFIRMED_FIXED` na gałęzi | Workflow oraz `scripts/quality/semgrep_gate.py`; negatywne testy wykazały wcześniejsze tolerowanie niekompletnego skanu | [#3286](https://github.com/Dudzian/Dudzian/pull/3286), 43 testy | Rzeczywiste skany nadal niekompletne; odrębne CH-SEC-08 |
| CH-SEC-02 | Medium / raporty CI i operatora / medium | `CONFIRMED_FIXED` na gałęzi | `scripts/enforce_coverage.py`, `deployment/core_test_plan.py`; 42 testy nie przeszły przed poprawką | [#3283](https://github.com/Dudzian/Dudzian/pull/3283), 101 testów i 393 sąsiednie | Naprawa odmowy DTD i limitu odczytu; nie jest dowodem wcześniejszej eksfiltracji XXE |
| CH-SEC-03 | High / lokalny cloud gRPC / medium | `CONFIRMED_FIXED` na gałęzi | Cloud manager, RPC, klient i lifecycle; negatywne testy świeżości oraz pustej wymaganej konfiguracji | [#3288](https://github.com/Dudzian/Dudzian/pull/3288), 48 testów | Aktualizacja protokołu musi objąć klienta, serwer i stuby; brak rozproszonego magazynu sesji |
| CH-SEC-04 | High / lokalny workflow commit / medium | `CONFIRMED_FIXED` na gałęzi | `scripts/quality/betterleaks_hook.py`; 5 negatywnych testów indeksu nie przeszło przed poprawką | [#3287](https://github.com/Dudzian/Dudzian/pull/3287), 63 testy i realny scanner | Binaria/limity odmawiają; hook nie chroni przed osobą mogącą zmienić jego konfigurację |
| CH-SEC-05 | Medium / lokalny import aktualizacji / medium | `CONFIRMED_FIXED` na gałęzi | `core/update/offline_updater.py`; 7 negatywnych przypadków awarii przed poprawką | [#3285](https://github.com/Dudzian/Dudzian/pull/3285), 26 testów | Nie obejmuje SIGKILL, power loss, współbieżności, automatycznego recovery ani całej instalacji |
| CH-SEC-06 | High / zależności aplikacji i build / zależna od advisory | Częściowo naprawione; `CONFIRMED_OPEN` dla narzędzi | `ui/desktop/package-lock.json`; aktualny pełny npm audit exit 1 | [#3289](https://github.com/Dudzian/Dudzian/pull/3289), 18 forms + 5 renderer E2E | Końcowo 4 pakiety high związane z GHSA-fc5q-39xc-vjpf; native Electron nietestowany |
| CH-SEC-07 | High / listener przy błędnej konfiguracji / medium | `CONFIRMED_FIXED` na gałęzi | Cztery konstruktory runtime gRPC i wspólny guard; negatywny baseline 84 FAIL | [#3290](https://github.com/Dudzian/Dudzian/pull/3290), 172 testy | Lokalny auth i rzeczywista kwalifikacja TLS/mTLS pozostają odrębne |
| CH-SEC-08 | High / wiarygodność CI / potwierdzone wykonaniem | `CONFIRMED_OPEN` | Rzeczywiste raporty Semgrep 1.179.0: timeouty i błąd parsowania | #3286 poprawnie odmawia obu raportom | Niepełnego pokrycia nie utożsamiać z bezpiecznym kodem |
| CH-SEC-09 | Nieustalona; potencjalnie wysoka / historia publicznego Git / nieustalona | `POTENTIAL`, nierozstrzygnięte | Nowy skan pełnej historii `origin/main`: 84 kandydatów, exit 1 | Bez publikacji ścieżek, wartości i surowego raportu | Potrzebny prywatny przegląd i dowody rotacji/odwołania; brak PASS |
| CH-SEC-10 | Do ustalenia / end-to-end update i inne granice / nieustalona | `POTENTIAL` | Odrębne kontrakty konsumentów i zaufania wymagają weryfikacji | Brak publicznego proof i brak pozornej poprawki | Szczegóły wyłącznie w odpowiednio prywatnym kanale |

Brak oznaczenia `CONFIRMED_OPEN` w triage sygnałów community nie obejmuje
nieprzeanalizowanych celów ani nie dowodzi braku innych podatności.

## 4. Walidacja

Środowisko: Linux, CPython 3.12.14, Ruff 0.14.11, mypy 1.10.0,
Semgrep CE 1.179.0, Betterleaks 1.9.0. Archiwum Betterleaks zweryfikowano
SHA-256 z przypiętego workflow; nie używano walidacji sekretów u dostawców.
Mypy Linux/Win32 wykonano w izolowanym środowisku hooka repozytorium;
`--platform win32` jest analizą statyczną, nie natywnym testem Windows.

### Commity i wyniki komponentów

| Obszar | Commit kodu | HEAD draft po integracji main | Wyniki lokalne |
| --- | --- | --- | --- |
| XML | `b2e7a99df028d26ae22340f81f70ea5ff8a1a37f` | `815cbb6e0d81c3a6dd2def3601460d2521e40a5e` | 101 + 393 różne testy PASS; 100% changed lines; po integracji 140 PASS |
| Semgrep gate | `7c50af1146c3d7133e88d2cad90e1e7a9e472dac` | `03cca4b3f62df4dc70a494a2c31be20f510646db` | 43 PASS; coverage 94% z pomiarem gałęzi; po integracji 82 PASS |
| Auth | `2a15911a51c76474d0420e10f636799b486a7dc2` | `495ae69c01162e6b9010ebc185a4c1a5a8e5a2f6` | 48 PASS; diff coverage 91%; po integracji 87 PASS |
| Git index | `da715c6167fd667e26a51df84a3d3b5fa9a7d256` | `dc784fef8b9aca7c13ab73f913ca77b2ee2182ad` | 63 PASS i realne testy scanner/config/baseline; po integracji 102 PASS |
| Update | `dbdc1c595ee2be49aa183f2d0dc59db40b696dcd` | `c32237c52290d98807cdb10cd75b5b1929afe791` | 26 PASS; po integracji 65 PASS |
| npm / renderer | `1c60786e64a6aa94d33eed72e40d122a86556f87`, raport `b05492d6` | `2fcb635fc2cdeb83c26abaca36f67a6db770f36d` | 18 forms + 5 renderer E2E PASS; build/tsc PASS; kontrakt CI 39 PASS |
| Runtime gRPC | `af32e24d69442deecabd4177cebd8787299e4a6a` | `a9fbe5c236ab3e78ce58df2934a9616f17228236` | 172 PASS; 100% changed-code coverage (29/29); po integracji 211 PASS |

Liczby po integracji obejmują wspólny kontrakt CI; nie należy ich sumować
jako odrębnych testów. Wynik SKIP ani cancelled nie został policzony jako PASS.

Polecenia komponentów wykonano z odpowiedniego checkoutu, używając `python`
z przygotowanego venv. Dokładny format hook na każdym końcowym HEAD:
`PRE_COMMIT_HOME=/workspace/work/pre-commit python -m pre_commit run ruff-format --all-files`
zakończył się kodem 0. Ruff i `git diff --check` także zakończyły się kodem 0.

| Polecenie / zakres | SHA i exit | Dowód oraz granica |
| --- | --- | --- |
| `pytest --noconftest tests/test_enforce_coverage_script.py tests/deployment/test_core_test_plan.py -q` | XML code SHA, 0 | 101 PASS; baseline tych samych testów 42 FAIL / 59 PASS, exit 1 |
| `pytest --noconftest tests/scripts/test_s9_fv_01_junit.py tests/deployment/test_stage9_observer_kvm_domain_v5d1.py tests/scripts/test_platform_core_evidence_workflow.py -q` | XML code SHA, 0 | 393 sąsiednie PASS |
| `pytest --noconftest tests/quality/test_semgrep_gate.py -q` | Semgrep code SHA, 0 | 43 PASS, także odmowy braków, awarii i wycieku treści raportu |
| `pytest tests/cloud/test_cloud_auth_freshness.py tests/cloud/test_cloud_security_license.py tests/cloud/test_cloud_service_health.py tests/cloud/test_hwid_enforcement.py tests/runtime/test_cloud_client.py -q` | Auth code SHA, 0 | 48 PASS; coverage startuje przed importami, a istniejący cykl importów wymaga wcześniejszego importu `bot_core.strategies` |
| `pytest --noconftest tests/quality/test_index_secret_scan.py tests/quality/test_secret_hygiene.py -q` | Git index code SHA, 0 | 63 PASS; osobne realne repozytoria i syntetyczne kandydaty |
| `pytest -q tests/update/test_offline_updater.py tests/update/test_offline_update_flow.py tests/update/test_update_package_cli.py tests/test_offline_updates.py` | Update code SHA, 0 | 26 PASS |
| `pytest -p asyncio tests/security/test_grpc_bind_boundary.py -q` | Runtime gRPC code SHA, 0 | 172 PASS, także rzeczywiste listenery |
| `python -m mypy --config-file=pyproject.toml --platform linux` i odpowiednik `--platform win32` | Commity komponentów, 0 | Skonfigurowany scope 424–425 plików, bez poszerzania o całe legacy |
| Dodatkowy mypy trzech modułów auth | Auth code SHA, 1 | 6 istniejących błędów; identyczne na b143b570, 0 nowych; brak suppressions |
| `pytest -p asyncio` dla auth/authorization/session/PIN/secret store/HTTP URL/tokenów/risk engine | `b143b570`, 0 | 175 baseline PASS; bez uruchamiania operacji finansowych |
| `pytest -p asyncio tests/security/test_postgresql_preaccount_credential_contract.py tests/security/test_postgresql_freshness_authority_acl_contract.py tests/security/test_atomic_root_proof_commit_contract.py -m 'not hardware and not destructive and not network and not external_postgresql' -q` | Baseline source bez zmian w nowym main, 0 | 294 PASS; kontrakty, nie rzeczywista baza PG |
| `python -m pip_audit --requirement deploy/packaging/requirements-desktop.lock --format json` | Lockfile b143b570, 0 | 104 zależności, 0 znanych podatności w wykonaniu z 17 XI; osobny odświeżony wynik 18 XI poniżej |
| `npm run test:forms`, `npm run build`, `tsc --noEmit` dla obu konfiguracji | npm code tree, 0 | 18 PASS oraz build/typy |
| `PLAYWRIGHT_BROWSERS_PATH=... npm run test:e2e` | npm code tree, 0 | 5 PASS; Chromium i testowy desktopAPI, bez Electron native |
| `npm audit --json` / `npm audit --omit=dev --json` | Końcowy npm lock, 1 / 0 | 4 high w narzędziach / 0 w zależnościach produkcyjnych |
| `betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 --report-format json .` | Baseline b143b570, 0 | 78,89 MB working tree, bez znalezisk poza baseline; odrębne od historii |
| `betterleaks git --redact=100 --report-format json --log-opts=origin/main .` | Historia do `34629687`, 1 | 137,80 MB, 84 nierozstrzygnięte kandydaty; bez surowego artefaktu w repo |

### Semgrep: wykonanie, kompletność i triage

Rzeczywiste skany 1.179.0 zachowały istniejący scope `bot_core core deployment scripts`.
Community zachowuje istniejące wyłączenia tests/archive/ccxt. Wyniki na b143b570:

- Blocking: 0 findings, 4 timeouty i 4 wpisy niepełnej analizy; brak PASS.
- Community: 25 findings, 24 errors (23 Timeout, 1 PartialParsing), 1246 scanned,
  21 skipped, z czego 14 oznacza niepełną analizę; brak PASS.
- Aktualny gate odrzuca oba zachowane raporty kodem 2. Kod procesu skanera
  również nie był zerowy (blocking 2, community 3).
- Przejrzano 25/25 bieżących sygnałów w kodzie i callerach: 25 oceniono jako
  `FALSE_POSITIVE`, 0 `POTENTIAL`, 0 `CONFIRMED_OPEN` w tym konkretnym zbiorze.
  Wszystkie dotyczą active source, dziewięć narzędzi test-support.
  Nie dodano wyjątków reguł. Lokalizacje porównane z gałęzią po integracji:
  25/25 bez zmian. Nie jest to odzyskanie wcześniejszego raportu ani test
  nieprzeanalizowanych celów.

Pełne dowody triage i surowe logi pozostają poza publicznym repo. Publiczne
Actions otrzymują tylko liczniki i stałe kategorie. Gdy community odmawia,
późniejsze kroki joba mogą być SKIPPED; nie kwalifikuje to sekretów ani innych skanerów.

### Integracja poprawek i Actions

Lokalne drzewo integracyjne: `be9bc20f9b6a7a1db00a801ac6fbb37c8c23fc04`,
zbudowane z aktualnego main i jawnie przejrzanych commitów powyżej.
Konflikt workflow rozwiązano przez zachowanie obu nowych zestawów testów;
nie usunięto wcześniejszego zakresu. Nie utworzono zbiorczego PR z tym drzewem.

| Kontrola drzewa integracyjnego | Exit | Wynik |
| --- | --- | --- |
| Celowana regresja wszystkich siedmiu obszarów i property tests pod `coverage run --branch` | 0 | **569 PASS, 1 SKIP** w 51,31 s |
| Smoke z executorami jak w CI: `pytest -p asyncio -p pytest_cov.plugin -p xdist.plugin -n 4 tests/property tests/quality/test_semgrep_gate.py tests/security/test_grpc_bind_boundary.py`, z dotychczasowymi i nowymi scopes `--cov` | 0 | **224 PASS**, 5 ostrzeżeń w 54,73 s; to podzbiór poprzednich przypadków |
| `coverage xml` i `diff-cover --compare-branch=origin/main --fail-under=70` | 0 / 0 | **94% zmienionych linii**, 472/498 w 13 mierzonych modułach; próg 70% zachowany |
| `python -m pre_commit run --all-files` z repozytoryjnym `PRE_COMMIT_HOME` | 0 | Ruff, format, mypy Linux i staged Betterleaks hook PASS |
| `python -m mypy --platform win32` w izolowanym interpreterze hooka | 0 | 426 plików, wyłącznie istniejące uwagi annotation-unchecked |
| `ruff check bot_core core scripts tests deploy stage6_samples ui` | 0 | Pełny podstawowy scope bez findings |
| `betterleaks dir` z niezmienioną baseline, redakcją 100% i raportem JSON | 0 | 117,80 MB working tree, bez findings poza baseline |
| Odświeżony 18 XI `pip_audit --requirement deploy/packaging/requirements-desktop.lock --format json` | 0 | **104 zależności, 0 znanych podatności**, bez wyjątków |

Smoke obejmuje kształt wykonania i zmienione scopes, a nie cały wielogodzinny
job ani realne PostgreSQL. Staged hook przy czystym indeksie nie zastępuje
odrębnego skanu working tree. Pomiar diff coverage obejmuje wskazane moduły
produkcyjne; nie jest pomiarem pełnego legacy, testów, TypeScript ani
automatycznie generowanych stubów.

Środowisko testów integracyjnych dopasowano do wersji z kanonicznego lockfile:
NumPy 1.26.4, pandas 2.2.2, SciPy 1.17.1, PyArrow 25.0.1. `pip check` i import
runtime tych bibliotek zakończyły się sukcesem. Wcześniejsze próby z nowymi
nieprzypiętymi bibliotekami zatrzymały kolekcję na NumPy/PyArrow; zachowano je
jako nieudane próby środowiskowe, bez zaliczania ich jako testów produktu.
Nie zmieniano lockfile Pythona, konfiguracji mypy ani testów, aby ukryć te błędy.

Regresja 569 przypadków obejmuje dokładnie moduły komponentów wymienione wyżej,
`tests/runtime/test_market_data_service.py`, `tests/test_metrics_service.py`,
`tests/test_metrics_service_tls.py`, `tests/test_risk_service.py`,
`tests/cloud/test_cloud_transport_security.py`, `tests/scripts/test_stage9_safe_ci.py`
oraz `tests/property`. Uruchomiono ją z
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0`.
Coverage działało od początku procesu, a runner przed `pytest.main(args)`
wykonywał `import bot_core.strategies` dla istniejącego cyklu importów.

Actions dla końcowych HEAD sprawdzono 18 XI o 13:19 UTC: część kontroli
zakończyła się sukcesem, pozostałe były queued/in_progress/skipped/neutral.
Brak zakończonego pełnego zestawu nie jest PASS. Starsze CI XML dla 7ded473e
miało 22308 PASS / 1 FAIL kontraktu inventory; naprawia go już scalony #3284.
Nie przypisano tamtych wyników nowym HEAD. Aktualny snapshot i oddzielne
wyniki kwalifikacji wymagają ponownego odczytu przed decyzją o merge.

| PR | success | queued | in_progress | skipped | neutral | failure |
| --- | --- | --- | --- | --- | --- | --- |
| #3283 | 20 | 18 | 2 | 4 | 1 | 0 |
| #3285 | 15 | 43 | 5 | 6 | 1 | 0 |
| #3286 | 14 | 26 | 0 | 4 | 0 | 0 |
| #3287 | 1 | 38 | 0 | 4 | 0 | 0 |
| #3288 | 1 | 36 | 2 | 4 | 0 | 0 |
| #3289 | 1 | 38 | 0 | 4 | 0 | 0 |
| #3290 | 1 | 38 | 0 | 4 | 0 | 0 |

To snapshot check-runs pobranych z paginacją, nie lista formalnie wymaganych
branch-protection checks. CodeQL, SBOM i pozostałych nieukończonych kontroli
nie zaliczono na podstawie wyników poprzednich SHA.

## 5. Pokrycie obowiązkowych 14 obszarów

| Obszar | Ocena i dowód | Luka pokrycia |
| --- | --- | --- |
| 1. Entrypointy | Mapa powyżej, bez mieszania archiwum i demo | `NOT_TESTED`: kompletne wykonanie wszystkich launcherów i deploymentów |
| 2. Authentication / sesje | `CONFIRMED_FIXED` w #3288; istniejące semantyczne sesje mają regresje | `NOT_TESTED`: wieloprocesowość i całe wdrożenie |
| 3. Authorization / RBAC / IDOR | Sprawdzono token scopes oraz account/operator/device binding w istniejących testach | `NOT_TESTED`: pełna macierz metod/obiektów i izolacja wielu klientów; brak deklaracji pełnej izolacji tenantów |
| 4. Klucze API / sekrety | Przegląd natywnego storage, testy syntetyczne, poprawka indeksu | `NOT_TESTED`: realny keychain, dumpy procesu, wszystkie logi i artefakty; historia nierozstrzygnięta |
| 5. gRPC / HTTP / TLS | `CONFIRMED_FIXED` plaintext guard w #3290; cloud guard już `MERGED` | `NOT_TESTED`: produkcyjny TLS/mTLS, certyfikaty, trust anchors i cała sieć |
| 6. PostgreSQL / ACL / migracje | Przegląd wybranych zapytań parametryzowanych, Identifier, search_path i transakcji; 294 kontrakty PASS | `NOT_TESTED`: realne PG16, role/DAC/peer, restart/commit ambiguity i fizyczne backupy |
| 7. Input / ścieżki / deserializacja | `CONFIRMED_FIXED` wybrane XML i indeks; testy URL i formularzy | `POTENTIAL` / `NOT_TESTED`: wszystkie archiwa, instalatory, SSRF i deserializatory end-to-end |
| 8. FastAPI / HTTP security | Odczytowa fabryka risk dashboard; Query limit i SQLAlchemy bind w źródle | `NOT_APPLICABLE` dla cookies/CSRF w stdin/gRPC; `NOT_TESTED` dla osobnego HTTP deploymentu/CORS/CSP/auth |
| 9. Rate limiting | Nowy bounded throttling auth; istniejący bucket gateway | `NOT_TESTED`: komplet metod uprzywilejowanych, ruch per źródło i limity usług w deployment |
| 10. Zależności / lock / SBOM | pip-audit i świeży npm audit; poprawka npm | `CONFIRMED_OPEN`: narzędziowe npm; SBOM/Actions końcowych SHA oczekują |
| 11. CI / skanery / sekrety | `CONFIRMED_FIXED` odmowa niepełnego Semgrep i staged snapshot | `CONFIRMED_OPEN`: niekompletne skany; historia niezaliczona, CodeQL oczekuje |
| 12. Hasła / PIN | Testy lockout, generation fences i bezpiecznej obsługi błędów; rozdzielono PIN od szyfrowania sekretów | `NOT_TESTED`: rzeczywisty adapter KDF/comparator/provisioning PIN i jego trwałe wdrożenie |
| 13. Operacje finansowe / risk gate | Źródła sprawdzają wartości finite i granice; istniejące testy oraz property tests | `NOT_TESTED`: cały łańcuch live i realne giełdy; brak autoryzacji handlu live |
| 14. Threat model / dowody / ryzyko | Ten raport, osobne PR-y i checkpoint | Braki jawnie wymienione; issue pozostaje otwarte |

## 6. Nierozwiązane ustalenia i prywatne dowody

1. Domknąć błędy Semgrep bez zmniejszania scope, ukrywania błędów ani uznawania
   `NO_FINDINGS` za bezpieczeństwo. Triage 25 sygnałów tego nie zastępuje.
2. Rozstrzygnąć 84 kandydatów historii w prywatnym przeglądzie. Podział reguł:
   41 generic-password, 28 generic-api-key, 13 private-key, 2 telegram-bot-api-token.
   To liczby kandydatów, nie 84 potwierdzone aktywne sekrety. Nie publikować wartości,
   ścieżek ani procedur nadużycia; potrzebne dowody odwołania/rotacji tam, gdzie właściwe.
3. Pełny npm audit pozostaje FAIL. Publiczny advisory:
   [GHSA-fc5q-39xc-vjpf](https://github.com/advisories/GHSA-fc5q-39xc-vjpf).
   Nie znaleziono poprawionej wersji picomatch ponad dostępną 4.0.7.
4. Zweryfikować end-to-end update i granice uprawnień w odrębnych testach;
   nie utożsamiać odzyskiwania pojedynczego importu z bezpieczeństwem instalacji.
5. Przed wdrożeniem wykonać właściwe natywne i fizyczne kwalifikacje oraz
   wymagane kontrole CI. Ich brak nie jest defektem, który wolno obejść.

`SECURITY_ADVISORIES=UNAVAILABLE` dla dostępnego konektora: brak operacji
zarządzania advisory. Jednorazowy odczyt REST listy był dostępny i pusty;
nie dowodzi to uprawnienia zapisu. Nie eksperymentowano z tokenami ani nie
obchodzono ograniczeń. Pełne nierozwiązane dowody należy przekazać właścicielowi
w prywatnym GitHub Security Advisory lub innym uzgodnionym prywatnym kanale.
Zwykłe issue, publiczny PR i artefakt Actions nie są takim kanałem.

## 7. Gotowość do scalenia

| PR | Ocena |
| --- | --- |
| #3283 XML | Draft; komponent i integracja lokalna zaliczone, aktualne wymagane Actions nieukończone |
| #3285 update | Draft; wąska poprawka zaliczona lokalnie, jawne ograniczenia i oczekujące Actions |
| #3286 Semgrep | Draft; gate działa, ale rzeczywista niekompletność skanów blokuje zaliczenie joba |
| #3287 indeks Git | Draft; testy i realny scanner zaliczone; wymagana kwalifikacja CI/platform i przegląd tradeoff binariów |
| #3288 auth | Draft; protokół wymaga skoordynowanej aktualizacji, istniejące targeted mypy errors jawne, Actions oczekują |
| #3289 npm | Draft; pełny audit FAIL i native Electron NOT_TESTED blokują gotowość |
| #3290 runtime gRPC | Draft; lokalne testy i coverage zaliczone, wymagane Actions i kwalifikacja deploymentu oczekują |

Nie rekomenduje się automatycznego merge żadnego PR. Osobne wymagane bramki
Linux/Windows pozostają aktywne i nie zostały osłabione.

## 8. Gotowość produkcyjna

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```

Wyniki jednostkowe, statyczny Win32, Chromium E2E i brak findings nie stanowią
kwalifikacji produkcyjnej. Risk gate, blokady operatora, podpisy, licencje,
ograniczenia handlu live i ręczne bramki pozostają bez zmian.

## 9. Punkt wznowienia

Źródła siedmiu poprawek są zapisane na zdalnych gałęziach wskazanych PR-ów.
Aktualny indeks postępu: `docs/security/security-audit-progress.md` na gałęzi
`security/audit-checkpoint-20261117`. Najpierw odczytać wymagane checks każdego
końcowego SHA, potem domknąć CH-SEC-08/09 i pozostałe braki, zachowując osobne
PR-y. Nie rozpoczynać audytu od nowa i nie przypisywać wyników do innego SHA.
