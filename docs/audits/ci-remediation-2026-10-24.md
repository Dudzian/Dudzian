# CryptoHunter — remediacja CI, 24 października 2026

## A. Root causes

Analiza obejmuje wszystkie **80 jobów** dwóch wskazanych przebiegów:
[40080330799](https://github.com/Dudzian/Dudzian/actions/runs/40080330799) i
[40080304463](https://github.com/Dudzian/Dudzian/actions/runs/40080304463),
z paginacją i odczytem logów wszystkich 74 wykonanych jobów. Pełny inwentarz,
wyniki, liczby warningów i pominięć znajdują się w
[audycie stanu wyjściowego](ci-remediation-2026-10-24-baseline.md).

[PR #3204](https://github.com/Dudzian/Dudzian/pull/3204), commit
`2a14d8db54c0ed506b1734e5f8765da21445eb66`, dodaje wyłącznie 56 linii Markdown.
Nie powoduje opisanych usterek. Osobna gałąź naprawcza wychodzi z aktualnego
`main`, pierwotnie `bae4fcb661de1de91f112347ac07f4220ed7e346`. Po scaleniu PR #3204
przez istniejący proces repozytorium bazę przesunięto fast-forward do
`aa625cd8f50453fdb6818000b7854f4f8568b892`; jedyna nowa zmiana bazowa to ta
sama notatka dowodowa. Ten PR jej nie zmienia.

| Problem | Przyczyna | Naprawa i dowód regresji |
| --- | --- | --- |
| Oba skany zależności FAILURE | Lock zachowuje podatny `setuptools==84.0.0`; generator nie wymuszał jego aktualizacji | Jawny pin 84.1.0 w generatorze, kanoniczna regeneracja; pełny audyt bez wyjątków |
| AnyIO exception po zakończeniu pytest | LIFO `atexit`: portal zamykał się przed cache HTTP; późniejsze `client.aclose` tworzyło nowy portal podczas `atexit`. Nowy callback nie wykonywał się w tej samej fazie zakończenia | Rejestracja portalu raz oraz cleanup HTTP po imporcie portalu; rzeczywiste subprocessy potwierdzają zamknięcie klienta, portalu i jego wątku |
| Wyścig `run_sync`/shutdown | Portal był pobierany bez rezerwacji czasu jego użycia; zamknięcie mogło wyprzedzić `portal.call` | Licznik przyjętych wywołań pod Condition; shutdown czeka na ich zakończenie, nowe zgłoszenia podczas drain otrzymują jawny RuntimeError |
| Deadlock przy zagnieżdżonym workerze | Oczekiwanie nowych zgłoszeń na koniec shutdown tworzyłoby cykl parent coroutine → worker → shutdown → parent | Odmowa nowych zgłoszeń i shutdown z aktywnego potomka, bez automatycznego ponawiania operacji; dwa ograniczone subprocessy regresyjne |
| PostgreSQL `SET LOCAL` poza transakcją | Konstruktor atomic commit oraz kwalifikacja admina w `approve_prepared` używały połączenia autocommit poza transakcją | Jawna transakcja konstruktora; kwalifikacja admina w tej samej SERIALIZABLE co approval; guard odmawia IDLE autocommit przed SQL |
| 11 collection warnings | Importowane klasy wyjątków zaczynające się od `Test` trafiały do kolektora pytest | Sześć lokalnych `__test__ = False`; publiczne nazwy i zachowanie bez zmian; identyczne 4644 uporządkowane node IDs w porównaniu obejmującym same markery |
| 27 NumPy warnings na platformę | ATR/EMA warmup lub zerowy ATR daje całkowicie pusty zbiór dodatnich obserwacji; pandas median wywołuje NumPy nanmean | Median tylko z dodatnich skończonych obserwacji; brak obserwacji i nieważny bieżący bar dają neutralny kierunek, bez wymyślania zmienności |
| Ruff `WPS433` | Pozostałości po innym linterze | Usunięcie trzech nieobsługiwanych komentarzy, bez zmiany konfiguracji Ruff |
| Trzy rzeczywiste testy QML pominięte po błędach | Wiązania właściwości do własnej właściwości potomka oraz cykl kontrolera diagnostycznego; loader zamieniał warning na skip | Jawne przekazywanie zależności i błąd testu przy niepoprawnym ładowaniu zamiast skip |
| Zbyt szeroka ekspozycja testowego PostgreSQL | Mapowania Docker bez IP publikowały port na wildcard hosta | Wszystkie 14 mapowań usług na 127.0.0.1; rzeczywista odmowa po adresach hosta spoza loopback |

## B. Implemented fixes

### Zależności i odtwarzalność

[GHSA-xmj3-9gf7-h52w](https://github.com/advisories/GHSA-xmj3-9gf7-h52w)
dotyczy path traversal przez kontrolowane metadane namespace wheel w starszej
ścieżce Windows `install_as_egg`, zakres `>=38.2.2,<84.1.0`, CWE-22.
Minimalna poprawiona wersja to 84.1.0; dostępna nowsza 84.2.1 nie jest potrzebna
do tej naprawy. Ograniczenia briefcase, ccxt, PyInstaller i hooks dopuszczają
84.1.0. Wheel `py3-none-any`, Python >=3.10, jest zgodny metadanymi z Python 3.11
na Linux, Windows i macOS; nie zastępuje to testów natywnych.

Uruchomiono `scripts/compile_desktop_lock.sh` na Python 3.11.16, pip-tools 7.5.2.
Zmiana locka to **jedna wersja**, bez dodawania/usuwania pakietów. Drugie
uruchomienie daje identyczny SHA256:
`ccfe09a5d808f7311b6fb07a864b8844279ca059a403228384cdb0b84ddb567c`.
Zweryfikowany wheel odrzuca niepoprawne namespace, ścieżki Windows/UNC/POSIX,
parent traversal i wyjście przez istniejący symlink; poprawny namespace działa.

Wyjątek joblib został przeanalizowany oddzielnie:
[oficjalny PYSEC-2024-277](https://github.com/pypa/advisory-database/blob/main/vulns/joblib/PYSEC-2024-277.yaml)
jest wycofany od `2026-06-09T16:03:00Z`. Po rzeczywistym audycie całego locka bez
wyjątków usunięto stary `--ignore-vuln`. Nie oznacza to, że wolno deserializować
niezaufany pickle/joblib. Kontrakt workflow wymaga braku ignorowania advisory,
`--fix` i `continue-on-error`, zachowując upload raportu po awarii.

### Portal, obliczenia i jakość testów

Potwierdzono brak skuteczności ustawienia lokalnego i oddzielenie kwalifikacji od
transakcji approval; nie stwierdzono produkcyjnej eskalacji uprawnień. Moduł
pozostaje ISOLATED_TEST_ONLY, a większość zapytań kwalifikuje nazwy katalogów.

Przyjęte wywołania portalu kończą się przed zamknięciem. Nowe wywołanie w trakcie
zamykania otrzymuje `RuntimeError` przed uruchomieniem funkcji asynchronicznej;
po zakończeniu shutdown można utworzyć nowy portal. Nie ponawiamy automatycznie
operacji, które mogą mieć skutki uboczne. Nie ma globalnego przechwytywania
wyjątków finalizacji. Testy zachowują contextvars i propagację błędów.

Neutralne `0.0` dotyczy **sygnału kierunku** strategii, a nie wartości ATR lub
zmienności. Nie uzupełniamy pomiarów zerami. Ważne obserwacje zachowują dotychczasową
skalę i bias; nieważny bieżący ATR nie może odziedziczyć wcześniejszej zmienności
i wyemitować kierunku. Testy bezpośrednie pluginu są konieczne: istniejący engine
przechwytuje wyjątek pluginu, więc samo `-W error` na wysokim poziomie wcześniej
również mogło zakończyć się PASS, pomijając sygnał.

Do 20 poleceń pytest dodano `-ra`, aby nowe logi pokazywały przyczyny pominięć;
usunięto też 10 istniejących `--disable-warnings`, aby audyt widział treść warningów;
nie zmieniono list testów, markerów, shardów ani timeoutów. Nie użyto `skip`, `xfail`, globalnego filtra warningów ani nowego wyjątku skanera
jako zamiennika naprawy. Warunek opcjonalnego testu przeglądarkowego przeniesiono
z poziomu całego modułu do osobnego scenariusza, aby zawsze wykonywać QML.

### Dodatkowe luki w wiarygodności UI

Usunięto 11 obserwowanych pętli wiązań w MainWindow/ModeWizard, używając jawnych
referencji do korzenia i istniejących kontrolerów kontekstu. Uaktywnienie tych
wiązań odsłoniło kolejne błędy DockManager: własność delegata przesłaniała id
korzenia oraz brakowało jawnego `required property var modelData`. Poprawka
zachowuje zawartość i kolejność paneli; test rzeczywistego Qt sprawdza przekazane
kontrolery, wypełnione delegaty i brak wszystkich ostrzeżeń. Testy zapisują stan
w tmp_path i zwalniają engine również przy niepowodzeniu.

Brak FastAPI był nieuzasadnioną luką pokrycia endpointu. Wyłącznie extra `test`
otrzymuje `fastapi==0.135.1` i `starlette==1.10.0`: nowszy FastAPI wymaga
Pydantic niezgodnego z kanonicznym 2.7.4. Audyt dokładnych 14 zależności przechodzi.
Test używa istniejącego `httpx==0.28.1` przez rzeczywisty ASGITransport, zachowując
asercje statusu i danych. Nie importuje deprecated Starlette TestClient i nie
dodaje drugiego stosu HTTP ani aktualizacji idna. Preflight trzech jobów QML jawnie wymaga FastAPI przez `--require-fastapi`,
a negatywne testy sprawdzają exit 1 przy brakujących FastAPI/httpx. Domyślne
wywołania innych jobów, również instalujących wyłącznie extra dev, zachowują
dotychczasowe wymagania. Desktop lock,
jego hash oraz runtime pozostają bez tej dodatkowej zależności.

### PostgreSQL: bezpieczeństwo i recovery

Kwalifikacja admina następuje po `SET TRANSACTION ISOLATION LEVEL SERIALIZABLE`
i pozostaje w tej samej transakcji co `approve_commit`. Domyślne połączenia
psycopg mogą nadal rozpocząć transakcję implicit; autocommit wymaga INTRANS.
`SET LOCAL` zachowuje zakres transakcji/savepoint, role i `pg_catalog` wracają
do ustawień sesji po zakończeniu. Nie zastąpiono go przez sesyjne `SET`.

W jobie [120833577103](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577103)
kolejność zdarzeń PostgreSQL odpowiada konkretnym testom:

1. `10:13:08` administrator terminates backend: test rollbacku przed COMMIT,
   porównujący pełny fizyczny snapshot z początkiem.
2. `10:13:10` fast shutdown: kontrolowany restart i sprawdzenie dokładnych bajtów,
   historii oraz idempotentnego retry.
3. `10:13:39` SIGKILL backendu na cut `before_commit`, reinitialization i recovery:
   test `test_abrupt_backend_sigkill_before_commit_recovers_no_partial_write_set`.
   Zachowuje pełne asercje braku częściowego write set i indeterminate lookup;
   dodatkowo sprawdza `pg_is_in_recovery() = false` i poprawne oryginalne retry.
4. WAL redo `Database/DROP: dir 1663/20188` wiąże komunikaty `base/20188` z replay
   usunięcia jednorazowej bazy użytej w poprzedzającym teście odmowy innej fizycznej
   tożsamości. Historyczny log nie drukuje losowej nazwy bazy, więc dokładnego
   mapowania OID/nazwa nie odtwarzamy jako faktu. Nie jest to dowód utraty danych
   aplikacji ani dowód braku każdego osieroconego pliku; kontener jest usuwany.
5. Późniejsze trzy restarty odpowiadają zachowaniu pending close i local completion
   przed/po każdym trwałym punkcie lifecycle.

Lokalny PostgreSQL 16.15 odtworzył ten sam komunikat WAL dla `base/19140`.
Po recovery sprawdzono brak OID w `pg_database` i brak katalogu przez
`pg_stat_file(..., true)`. Po pełnych testach oba klastry miały zero tymczasowych
schematów i dodatkowych baz; własne kontenery i wolumeny zostały usunięte.

Drugi wskazany job powtarza te scenariusze, z SIGKILL/recovery około `10:15:58`.
Oczekiwane błędy NOLOGIN, privilege/CAS i administrative disconnect zachowują
negatywne asercje. Nie usunięto fault injection ani testów awaryjnych.

Testowe `trust` pozostaje dla świeżych passwordless LOGIN roles sprawdzanych jako
rzeczywisty `session_user`. Samo przełączenie na SCRAM zatrzymałoby testy przed
asercjami ACL. Loopback ogranicza publikację portu; nie uwierzytelnia procesów
runnera ani innych kontenerów tej samej sieci. Dokładne granice i oddzielny dowód
native peer authentication opisano w
[modelu izolacji PostgreSQL CI](../security/ci-postgresql-isolation.md).

## C. Regression validation

Wyniki lokalne dotyczą Linux; `mypy --platform win32` jest analizą statyczną,
nie uruchomieniem Windows. Nie przenosimy PASS z poprzedniego commita.

| Kontrola | Wynik lokalny |
| --- | --- |
| Pełny pip-audit, Python 3.11, bez ignorowania advisory | 104 pakiety, 0 podatności, exit 0 |
| Powtórna kanoniczna generacja locka | Identyczny SHA256 |
| Rzeczywiste rozwiązywanie `.[desktop]` względem locka na Linux | PASS |
| Regresje wheelhouse, zależności i kontraktów workflow na końcowym drzewie | 51 PASS |
| Portal AnyIO: 12 nowych + 24 istniejące testy HTTP/cache/script | 36 PASS |
| Stary kod portalu: rzeczywiste zakończenie procesu | 4/4 nowych przypadków zawodzą, zapisują wyciek portalu |
| Bezpośredni plugin + native auto-trade + katalog strategii | 37 PASS, RuntimeWarning jako błąd |
| Kolekcja security przy samej zmianie markerów | 4644 identyczne node IDs, 11 → 0 warningów |
| PostgreSQL 16.15: końcowe transakcje/guard/kontrakt/odmowa approval | 246 PASS, 0 pominięć; zakres częściowo pokrywa pełny moduł atomic |
| QML Qt 6.7.2: trzy ukryte scenariusze oraz frontend/bootstrap | 9 PASS, 0 pominięć, 0 warningów |
| Wcześniej pomijany endpoint FastAPI przez rzeczywisty transport ASGI | 1 PASS, wszystkie warningi jako błędy |
| Preflight i deklaracje zależności endpointu | 8 PASS, w tym odmowy brakujących FastAPI/httpx |
| Audyt dokładnego stosu testowego FastAPI | 14 pakietów, 0 podatności, bez ignorowania advisory |
| Kontrakty workflow i rzeczywiste odmowy SQL ACL/SET ROLE | 7 PASS |
| Mypy skonfigurowany scope Linux i `--platform win32` | Oba PASS, po 347 plików |
| Pełny atomic commit na PostgreSQL 16.15, wraz z SIGKILL/recovery | 60 PASS, 745.40 s |
| Dokładny hook `python -m pre_commit run ruff-format --all-files` | PASS na końcowym drzewie |
| Ruff baseline całego utrzymywanego zakresu | PASS |
| Betterleaks 1.9.0, offline, wszystkie pliki planowanego commita | PASS, brak wycieków |

Końcowe 246 PASS obejmuje najnowszą asercję braku wykonania SQL `approve_commit`
po odmowie kwalifikacji oraz domyślną implicit transaction; nie sumujemy tych
przypadków z 60 jako rozłącznych testów. Nowe workflow są oceniane oddzielnie
przed uznaniem PR za gotowy. Przerwane lokalne próby przy przeciążeniu limitu 2 CPU nie są PASS; testów ani
deadline nie osłabiono. Dodatkowy przebieg czterech modułów fixture dał 53 PASS
i 1 FAIL przy 15-sekundowym oczekiwaniu na pierwszy sygnał potomka. Ten dokładny
przypadek uruchomiony osobno z ograniczeniem wątków BLAS/OMP dał PASS (20.24 s
całego testu); wcześniejszy analogiczny source-lease dał osobno PASS (18.08 s).
Nie przepisujemy nieudanego całego przebiegu na PASS.

## D. Warning audit

| Kategoria | Przed | Po: potwierdzony zakres |
| --- | --- | --- |
| Zależności | 1 niezaakceptowana podatność | 0 w rzeczywistym pełnym skanie locka |
| NumPy | 27 w `test_auto_trade_engine_native.py` | 0; 18 PASS w tej samej konfiguracji NumPy 1.26.4/pandas 2.2.2 |
| Collection | 11 w kolekcji security | 0, identyczne node IDs; nowe regresje dodają testy |
| AnyIO | 3 wyjątki shutdown na przebieg CI | 0 w 36 lokalnych testach, w tym rzeczywistych subprocessach; pełne platformy wymagają nowego CI |
| SET LOCAL | 42 na przebieg CI, w tym 26 w każdym wyróżnionym jobie | 0 w celowanych testach oraz logu serwera po pełnych 60 testach atomic; pełne shardy wymagają nowego CI |
| Ruff WPS433 | 3 lokalizacje | Usunięte; konfiguracja bez zmian |
| QML błędne ładowanie zamieniane w skip | 3 pominięte scenariusze | 4 PASS bez warningów, oba wejścia TicketDialog zachowane |
| Frontend QML | 11 lokalizacji pętli wiązań | Usunięte; pełna grupa 9 PASS, pusty zbiór warningów również dla wypełnionych delegatów DockManager |
| FastAPI | 1 skip przez brak zależności | 1 PASS, bez skipów i warningów |
| Node DEP0005 | 29 wiadomości na każdy przebieg | Pozostaje zewnętrzny problem bez wydanej poprawki; brak wyciszenia |
| WAL Database/DROP directory | 2 powiązane wiadomości po SIGKILL | Oczekiwany scenariusz replay opisany powyżej; testy zachowane |

Node: oficjalny `download-artifact` v8.0.2, SHA
`9000827ccba6bdab643e8b6fd33ac0654aef8333`, używa
`@actions/artifact@6.3.1` → `unzip-stream@0.3.4` → `new Buffer('')`.
Stos odtworzono przez `node --trace-deprecation` na parserze ZIP oficjalnego bundle.
[v8.0.2](https://github.com/actions/download-artifact/releases/tag/v8.0.2)
i [upload v7.0.2](https://github.com/actions/upload-artifact/releases/tag/v7.0.2)
są aktualnymi wydaniami i nadal zawierają tę implementację. Nie wprowadzono
forka ani tłumienia deprecations. Wszystkie 29 par digestów w każdym historycznym
przebiegu pasuje; `digest-mismatch: error` i sprawdzanie integralności pozostają.

## E. Security assessment

Nie zmieniono frozen contracts, zaufanych manifestów/trust pins, polityki CodeQL,
progów Quality Ratchet, grantów produkcyjnych ani protokołu transakcyjnego A+C.
Zachowano `stage_9=IN_PROGRESS`, `production_provisioning_ready=false`,
`private_control_status=BLOCKED`, `ROOT_PROOF_ISSUANCE_AVAILABLE=false`,
produkcyjne SIGN/BIND BLOCKED i Stage 10 NO-GO. Nie wykonywano provisioningu,
enrollmentu, aktywacji ani fizycznego Stage 41B `--execute`.

`STAGE41B_TEST_ONLY_PASS` pozostaje wynikiem operator-reported dla syntetycznych
signerów na jednym Dellu, bez niezależnej atestacji. Hostowane testy macOS nie
kwalifikują fizycznego Maca. Odmowy platformowe `MISSING_CURRENT_EVIDENCE` oraz
`PRODUCTION_READY=NOT_READY` pozostają rzeczywistymi blokadami bezpieczeństwa.

## F. Remaining blockers

- Oryginalne `pip-audit-report.json` i historyczne JUnit: narzędzie GitHub zwraca
  ZIP jako File Service URL, którego host polityka sieci odmawia (403). Metadane
  artefaktów są odczytane, ich bajty nie. Lokalny raport bazowy odtworzono z
  oryginalnego locka i potwierdzono dokładnie advisory setuptools; nie nazywamy
  go pobranym historycznym artefaktem.
- Artefakty audytu: run 40080330799 ID 12554498121, run 40080304463 ID 12555256367,
  oba 2590 B. Brak dostępu do bajtów ogranicza niezależne potwierdzenie zawartości
  historycznych raportów oraz kompletnej klasyfikacji dawnych skipów.
- Historyczne logi głównych shardów nie drukują wszystkich przyczyn skip.
  Zidentyfikowane shardy i platformowe podziały są poprawne, ale nie uznajemy
  automatycznie wszystkich 122/1786/1596 pominięć za uzasadnione. Nowe `-ra`
  pozwoli ocenić je na aktualnym commicie.
- Historyczne pominięcie Playwright ukrywało cały test QML Workbench. Rozdzielono
  scenariusz Qt i opcjonalny browser mirror, zachowując wszystkie 20 oryginalnych
  asercji i teardown w finally. Qt 6.7.2 ujawnia rzeczywisty FAIL: brak
  `instrumentDetails`, potem również starszych `controlState`/metod schedulera
  i `maxExposureUtilization`. W merge `d16d7b8f` wybrano starszy monolityczny
  widok/model, pozostawiając test oraz nieużywane pliki paneli z `dc730e78`.
  Obecny widok nie wywołuje tych API. Przywracanie dawnego forwarding do
  rzeczywistego schedulera lub tworzenie atrap tylko dla testu byłoby nieuzasadnioną
  zmianą produktu. Zachowano widoczny FAIL jako blokadę decyzji o aktualnym
  kontrakcie widoku; Chromium pozostaje SKIP z powodu braku przeglądarki.
  Nie uznajemy żadnego z tych scenariuszy za PASS. Lukę FastAPI
  naprawiono i zweryfikowano oddzielnie, bez rozszerzenia desktop runtime.
- Upstream DEP0005 pozostaje do naprawy w wydanym, zweryfikowanym komponencie.
- Wykonanie Windows/macOS, CodeQL, Quality Ratchet i pełnych shardów PostgreSQL
  wymaga nowych rzeczywistych wyników CI. Lokalne wyniki ich nie zastępują.

## G. PR readiness

**NO-GO dla merge**, dopóki nie ma aktualnych wyników wymaganych workflow i
oceny pozostałych luk dowodowych. PR ma pozostać draft, jeżeli formatowanie
lub którakolwiek wymagana kontrola nie została zakończona. Nie wykonujemy merge.
