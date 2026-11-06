# S9-FV-01 P2 — obowiązkowe wykonanie prób migracji w PR #3258

6 listopada 2026 r. Oceniona baza PR: `d4ff6013e02ca220d404c219c1dfaaa8c9110b7b`.
Poprawka dotyczy wyłącznie egzekwowania testów w istniejącym Draft PR.

## Problem i wynik

Fixture `kernel_pidfd` celowo pomija próbę, gdy jądro lub sandbox nie udostępnia
`SO_PEERPIDFD`. To poprawna lokalna informacja o braku możliwości wykonania,
ale sam exit 0 pytest nie dowodził wykonania migracyjnej bramki CI. P2 zachowuje
fixture i testy transportu, a osobno wymaga rzeczywistego PASS wszystkich
obowiązkowych przypadków. Brak możliwości socketu, platformy lub PIDFD daje
**BLOCKED** przez skipped/error/brak pełnego raportu; nie jest PASS bramki.

## Dokładny wymagany zestaw

[Wykaz wszystkich 83 identyfikatorów, łącznie z parametrami](s9_fv_01_required_cases.md)
odpowiada [statycznemu manifestowi JSON](../../tests/fixtures/s9_fv_01_required_cases.json):

| Moduł | Wymagane przypadki |
| --- | --- |
| `tests.security.test_account_genesis_verified_transport_contract` | 46 |
| `tests.security.test_account_genesis_verified_transport_unix` | 33 |
| `tests.security.test_account_genesis_verified_transport_scope` | 4 |

Rzeczywista kolekcja pytest została porównana z tym wykazem oraz wcześniejszym
raportem 83 PASS. Zbiory są dokładnie równe. Manifest jest materiałem review;
CI nie tworzy go z ocenianego raportu ani nie odświeża go automatycznie.
Zmiana parametru, nazwy lub dodatkowy przypadek w obowiązkowym module wymaga
jawnego przeglądu wykazu. Brak pozycji nie jest uzupełniany sumą starszych PASS.

## Egzekwowanie w workflow

[Istniejący krok `quality-ratchet`](../../.github/workflows/quality-security.yml)
usuwa poprzedni plik `$RUNNER_TEMP/s9-fv-01-junit.xml`, następnie uruchamia pytest
z `junit_family=xunit2` i `--junitxml`. Zachowuje rzeczywisty exit pytest, także
1–5, i uruchamia [walidator](../../scripts/quality/check_s9_fv_01_junit.py) po każdym
zakończeniu polecenia. Nie używa `continue-on-error`. Wyjście pytest inne niż 0
nie jest maskowane przez obecność starego lub częściowego raportu.

Walidator wymaga dokładnej pary `(classname, name)` dla każdej pozycji —
również pełnego identyfikatora parametrów. Każda pozycja musi wystąpić dokładnie
raz jako zakończony PASS. Skipped, failure, error, notrun/disabled, brak,
duplikat albo nieznana pozycja obowiązkowego modułu oznaczają BLOCKED/exit 1.
Pusty, ucięty, nieczytelny, obcy XML, same liczniki lub niespójne liczniki
również nie dają PASS. Wykaz wymagany nie może być pusty ani zawierać duplikatów.
Parser używa biblioteki standardowej, ogranicza rozmiar wejścia i odmawia DTD.

Dozwolone skipped dotyczą wyłącznie pięciu starszych modułów regresji
z tego kroku: `test_freshness_runtime_ipc`, `test_freshness_runtime_ipc_identity`,
`test_freshness_verifier_transport_remediation`, `test_freshness_verifier_pg_remediation`,
`test_freshness_verifier_remediation_scope` (prefiks `tests.security.`).
Są liczone oddzielnie jako `starsze_SKIPPED`. Skipped nowych regresji walidatora
nie jest automatycznie zaliczany do starszego zakresu. Failure/error starszego
testu oraz niezerowy exit pytest nadal blokują cały krok.

JUnit jest dowodem wykonania testów w tym kroku, a nie niezależną atestacją hosta
czy kwalifikacją produkcji. Zaufanie do runnera i wykonanej wersji kodu pozostaje
istniejącą granicą CI.

## Próby dodatnie i ujemne

Regresje walidatora w [test_s9_fv_01_junit.py](../../tests/scripts/test_s9_fv_01_junit.py)
obejmują 8 dodatnich wariantów i negatywne warianty opisane niżej. Próby używają
syntetycznych XML oraz statycznego wykazu; nie wykonują administracyjnego PG.

| Próba | Wymagany wynik |
| --- | --- |
| Pełny dokładny wykaz; root testsuite/testsuites, dozwolone zagnieżdżenia | PASS |
| Pełne 83 oraz starszy skipped | PASS, osobny licznik starszych pominięć |
| Brak każdego wybranego required obok 100 starszych PASS | BLOCKED |
| Zmieniony parametr, klasa albo brak parametru | BLOCKED |
| Skipped z przyczyn PIDFD, socketu lub platformy | BLOCKED |
| Failure/error/skipped required; failure/error starszego testu | BLOCKED |
| Duplikat w jednym lub różnych suites; dodatkowy niezatwierdzony parametr | BLOCKED |
| Brak classname/name lub jawne notrun/disabled/skipped | BLOCKED |
| Pusty, ucięty, obcy lub niekompletny raport i niespójne liczniki | BLOCKED |
| Brak raportu, pusty/niepoprawny manifest | BLOCKED |
| Exit pytest 1–5, nawet obok kompletnego raportu | BLOCKED |
| Skipped nowego testu walidatora udający starszy zakres | BLOCKED |

Rzeczywista próba odtworzenia ograniczenia środowiska: pytest dla trzech modułów
migracji w sandboxie zwrócił **50 PASS / 33 SKIP i exit 0**; skipped wskazywały
`SO_PEERPIDFD`/EPERM. Ten sam raport ocenił walidator jako
**`S9-FV-01: BLOCKED — wymagane=83, PASS=50, starsze_SKIPPED=0`**, exit 1.
To bezpośrednio odtwarza i zamyka zgłoszony fałszywy sukces.

Rzeczywista próba dokładnego skryptu bash z workflow poza ograniczeniem sandboxa
zakończyła się **340 PASS / 1 SKIP** i wykonała wszystkie 83 obowiązkowe przypadki;
jedyne pominięcie starszego testu
wynikało z braku dodatkowych grup konta. Wynik bramki:
**`S9-FV-01: PASS — wymagane=83, PASS=83, starsze_SKIPPED=1`**.
AF_UNIX, SO_PEERCRED/PIDFD i procfs są rzeczywiste; instalacja/expected/PG/CAS
pozostają modelami dokładnie jak przed P2.

Końcowe nowe regresje P2: **73 PASS, zero SKIP/xfail**:
69 przypadków walidatora/wykazu (8 dodatnich, 61 ujemnych) oraz
4 przypadki integracji [literalnego bash z YAML](../../tests/scripts/test_s9_fv_01_junit_workflow.py)
(1 dodatni, 3 ujemne). Te cztery wykonują prawdziwy walidator, modelując tylko
wykonanie pytest: pełny raport → PASS; required skipped → BLOCKED; komplet
z exit pytest 2 → BLOCKED; stary poprawny raport i exit 5 bez nowego raportu
→ BLOCKED, z dowodem usunięcia starego pliku. Sprawdzają także, że walidator
został rzeczywiście wywołany z zachowanym kodem pytest.

Istniejące kontrakty CI bezpieczeństwa i jakości historii:
`test_ci_workflow_security_scan_contract.py` oraz
`test_history_reproposal_quality_ci.py`: **40 PASS**. Żadnego testu nie
wyłączono ani nie oznaczono xfail w celu uzyskania zielonego wyniku.

## Jakość i odtworzenie

| Kontrola | Wynik |
| --- | --- |
| Ruff 0.14.11 całego istniejącego zakresu oraz E/F/B/I trzech nowych plików Python | PASS |
| Mypy 1.10.0 Linux i `--platform win32` w skonfigurowanym zakresie | PASS, po 392 pliki |
| Jawne mypy nowego walidatora Linux/Win32 | PASS, po 1 plik |
| `python -m pre_commit run ruff-format --all-files` po dodaniu wszystkich plików do indeksu | PASS, exit 0, bez zmian |
| Betterleaks 1.9.0 offline, pełne repo i staged, dotychczasowy baseline i redakcja 100% | PASS, zero znalezisk |
| `git diff --check`, `git diff --cached --check` | PASS |

Pierwszą próbę odbudowy cache hooka ograniczał sandbox (proxy/EPERM).
Ten sam wymagany hook wykonano następnie z dostępem sieciowym przez zachowane
proxy; nie zmieniono polityki, narzędzi aplikacji ani locków. `--platform win32`
oznacza analizę typów na Linux, nie natywne wykonanie Windows.

```bash
python -m pytest -q tests/scripts/test_s9_fv_01_junit.py tests/scripts/test_s9_fv_01_junit_workflow.py
python scripts/quality/check_s9_fv_01_junit.py RAPORT_JUNIT.xml --pytest-exit-code KOD_PYTEST
python -m ruff check --select E,F,B,I scripts/quality/check_s9_fv_01_junit.py tests/scripts/test_s9_fv_01_junit.py tests/scripts/test_s9_fv_01_junit_workflow.py
python -m mypy
python -m mypy --platform win32
python -m mypy scripts/quality/check_s9_fv_01_junit.py
python -m mypy --platform win32 scripts/quality/check_s9_fv_01_junit.py
python -m pre_commit run ruff-format --all-files
```

Dokładny wspólny zestaw pytest i przekazanie kodu zakończenia są utrzymywane
jednorazowo w YAML, a test integracji wykonuje ten sam tekst. Ocena aktualnego
stanu GitHub Actions dotyczy konkretnego HEAD i jest raportowana po publikacji
aktualizacji PR; lokalnego PASS nie utożsamia się z zakończeniem zdalnego workflow.
Przed P2 dla ocenionego HEAD tylko Style preflight miał success, a siedem
pozostałych przebiegów było queued; nie uruchamiano ich ponownie.

## Zakres i kwalifikacje

Nie zmieniono istniejącego adaptera, trzech modułów testów migracji, loaderów,
obu propozycji patchy, produkcyjnego AccountGenesis/verifiera, obecnego K4,
pinów, podpisów, manifestów, kluczy, kontraktów ani konfiguracji/roli PG.
Nie obniżono progów CI. Nie utworzono nowego PR, nie wykonano merge ani restartu
workflow. Aktualizacja gałęzi może wyzwolić zwykłe nowe przebiegi pull_request.

Administracyjny Stage 9, PostgreSQL i natywne dwa UID pozostają odrębnymi
kwalifikacjami **NOT_RUN**. Wdrożenie **BLOCKED**, K4-v2 nieaktywny,
etap 1 **PENDING_OWNER_ACCEPT**, Stage 9 **IN_PROGRESS / PRODUCTION_NO_GO**.
