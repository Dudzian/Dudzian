# Alerty Code Scanning #59, #60 i #61: zakres dowodu

Aktualizacja: 2026-11-03. Raport rozdziela bieżącą poprawkę libpq, historię
kodu, ograniczenie ryzyka i dane rzeczywiście odczytane z GitHub.
Brak dostępu do Code Scanning API **nie jest blokerem merge poprawki w PR
[#3244](https://github.com/Dudzian/Dudzian/pull/3244)**. Nie stanowi też dowodu
naprawienia alertów ani false positive. W ramach tej poprawki nie ponawiamy
odczytów alertów #59–61, zmian tokenów ani uprawnień. Nie dodano suppressions.
PR nie został scalony ani nie zmieniono stanów alertów. Dowody CI nie
upoważniają do instalacji produkcyjnej, enrollmentu ani fizycznej kwalifikacji
Stage 9/10.

## Bieżąca poprawka: przypięcie endpointu libpq

Punktem wyjścia tej korekty jest HEAD
`3adba03d8803b5b869240c28bc70d3e31be79133`.
W [`_configure_database_with_password()`](../../deployment/windows_installer/provision.py)
jawne `host=127.0.0.1` nie wyłączało odziedziczonego `PGHOSTADDR`.
libpq mogło zachować nazwę hosta dla weryfikacji certyfikatu i dopasowania
passfile, a faktyczne połączenie TCP kierować na inny adres.

DSN przypina teraz `hostaddr=CONTRACT.postgresql_host`, czyli zatwierdzony
endpoint `127.0.0.1` z
[`contract.py`](../../deployment/windows_installer/contract.py), obok jawnych
`host`, `port`, `dbname`, `user`, `password=""`, chronionego `passfile`,
`sslmode=verify-full`, `sslrootcert` i `connect_timeout=5`.
`gssencmode=disable` zapobiega wyborowi szyfrowania GSS przed TLS przez libpq.
To zachowuje wymaganą weryfikację TLS i serwerową politykę SCRAM.
Ten sam DSN jest podstawą połączeń tworzenia bazy, provisioningu, kwalifikacji
i dowodu odmowy nowej sesji administratora po zapisaniu końcowego HBA.

Produkcja nie usuwa ani nie podmienia globalnych `os.environ`.
Takie operacje wpływałyby także na inne wątki i procesy potomne instalatora;
tymczasowe usunięcie i późniejsze przywrócenie nie usuwa tego wyścigu.
Jawne parametry są ograniczone do konkretnego połączenia i nie wymagają
szerszego refaktoringu.

### Zakres wpływu pozostałych zmiennych PG*

| Zmienne lub źródło | Ocena dla tego bootstrapu |
| --- | --- |
| `PGHOST`, `PGHOSTADDR`, `PGPORT`, `PGDATABASE`, `PGUSER` | Odpowiadające parametry są jawne. Nie wybierają innego endpointu ani użytkownika bootstrapu. |
| `PGPASSWORD`, `PGPASSFILE` | Jawne `password=""` blokuje fallback hasła ze środowiska; jawny `passfile` wskazuje plik SYSTEM-only. Pusty parametr nie pozwala obejść SCRAM. |
| `PGSSLMODE`, `PGSSLROOTCERT`, `PGCONNECT_TIMEOUT`, `PGGSSENCMODE` | Jawne `verify-full`, lokalna CA, timeout i `gssencmode=disable` mają pierwszeństwo. |
| `PGSERVICE`, `PGSERVICEFILE`, `PGSYSCONFDIR` | Mogą wybrać plik usługi i uzupełnić parametry niepodane w DSN. Nie nadpisują powyższych jawnych parametrów; błędna usługa może uniemożliwić połączenie. Plik usługi może także ustawić np. `sslkeylogfile` lub parametry kluczy SCRAM; przypięcie endpointu nie oznacza izolacji od wszystkich skutków wrogiej konfiguracji usługi. |
| `PGOPTIONS`, `PGAPPNAME`, `PGCLIENTENCODING`, `PGDATESTYLE`, `PGTZ` | Mogą zmieniać opcje startowe lub ustawienia sesji, a niektóre wartości powodować błąd. Nie są sanitowane. Nie zastępują przypiętego adresu, weryfikacji CA ani serwerowego HBA/SCRAM. |
| `PGTARGETSESSIONATTRS`, `PGLOADBALANCEHOSTS`, ustawienia dostępności obsługiwane przez daną wersję libpq | Mogą wpływać na akceptację sesji lub dostępność. Jawna pojedyncza para host/hostaddr nie daje im listy obcych endpointów do wyboru. |
| `PGSSLCERT`, `PGSSLKEY`, `PGSSLCERTMODE`, `PGSSLCRL`, `PGSSLCRLDIR` | Mogą wpływać na dodatkowe pliki i poświadczenia TLS albo spowodować odmowę. Nie zastępują jawnej zaufanej CA i `verify-full`. Zestaw ten nie jest deklaracją izolacji od wszystkich plików konfiguracyjnych środowiska. Parametr `sslpassword` nie ma zmiennej środowiskowej w sprawdzonym libpq 18.0.6. |
| `PGSSLSNI`, `PGSSLNEGOTIATION`, `PGSSLMINPROTOCOLVERSION`, `PGSSLMAXPROTOCOLVERSION`, `PGMINPROTOCOLVERSION`, `PGMAXPROTOCOLVERSION`, `PGCHANNELBINDING`, `PGREQUIREAUTH` | Zależnie od wersji libpq mogą zmieniać negocjację, ograniczenia protokołu lub dodatkowe wymagania uwierzytelnienia, także uniemożliwić połączenie. Nie wyłączają jawnego `verify-full` ani polityki SCRAM serwera. |
| `PGKRBSRVNAME`, `PGGSSLIB`, `PGGSSDELEGATION`, `PGREQUIREPEER` | Nie wybierają innego endpointu. Szyfrowanie GSS jest wyłączone, bootstrap HBA wymaga SCRAM, a połączenie Windows używa TCP. |

Poprawka izoluje wybór endpointu i wskazane parametry uwierzytelnienia;
nie obiecuje usunięcia wpływu każdego `PG*` na wszystkie nieokreślone
ustawienia sesji ani braku eksportu dowolnych sekretów przy dowolnej wrogiej
konfiguracji usługi. Nie rozszerzano zakresu na globalną sanitację środowiska.

### Walidacja nowej rewizji

Istniejący harness LocalSystem rozszerzono o błędne `PGPASSWORD`, obce
`PGHOSTADDR` oraz obie zmienne jednocześnie. Dowód obejmuje rzeczywisty
wybór endpointu przez libpq i połączenia do PostgreSQL, poprawny chroniony
passfile, odmowy błędnego hasła i certyfikatu, końcowe HBA oraz bezpieczne
stałe diagnostyki. Wykonanie nowej wersji na Windows pozostaje **PENDING**;
lokalne modele nie zastępują tego dowodu.

Lokalny zestaw ośmiu modułów Windows (bootstrap, diagnostyka, harness,
credentials, installer, role denial, backend recovery i service recovery):
**361 passed, 4 skipped**. Pominięcia dotyczą wyłącznie opt-in PostgreSQL
na Linux i natywnych ACL Windows. Rzeczywiste testy wyboru parametrów libpq:
**17 PASS**; nie są dowodem uwierzytelnienia LocalSystem. Pełny baseline Ruff,
lint i format zmienionych modułów oraz dokładny
`python -m pre_commit run ruff-format --all-files`: **PASS**.
Skonfigurowany mypy Linux i win32: **PASS**, po 387 plików.
Betterleaks: **PASS**, exit0, istniejący baseline, bez nowych suppressions.
Zestaw **361 passed, 4 skipped** powtórzono po zastąpieniu stałych
syntetycznych haseł losowymi wartościami fixture.

Nowy przebieg LocalSystem, MSI clean-install, CodeQL, Quality and Security
oraz pełne CI pozostają **PENDING** do odczytu wyników dla nowej rewizji.
Poniższe historyczne sukcesy nie stanowią PASS dla zmienionego kodu.
Brak fizycznych dowodów Linux i Stage 10 pozostaje odrębną sprawą i nie
blokuje tej poprawki libpq. `STAGE9_PRODUCTION_READY=NO` pozostaje zachowane.

## Historyczny dostęp do API

Ponowny odczyt ze skonfigurowanym `GITHUB_ALERT_TOKEN` 2026-11-03:

| Zakres | Czas UTC | Wynik |
| --- | --- | --- |
| `GET /repos/Dudzian/Dudzian/code-scanning/alerts/59`, `/60`, `/61` | 20:07:50 | Wszystkie HTTP 403: `Resource not accessible by integration` |
| Trzy odpowiadające endpointy `/alerts/{number}/instances` | 20:07:50 | Wszystkie HTTP 403: `Resource not accessible by integration` |
| `GET /repos/Dudzian/Dudzian/code-scanning/analyses` | 20:07:50 | HTTP 403: `Resource not accessible by integration` |

Wartości poświadczenia nie wypisano ani nie zapisano w raporcie. Historyczny
odczyt 2026-11-02 również zwracał 403; nagłówek wskazywał wymaganie
`security_events=read`. To ograniczenie autoryzacji, nie dowód stanu alertu.
Nie uzyskano `analysis_id`, więc powiązany SARIF i jego `codeFlows` pozostają
**nieodczytane**. Nie oznacza to braku analizy, przepływu ani zera wyników.

| Alert | Potwierdzony materiał | Aktualne metadane API |
| --- | --- | --- |
| [#59](https://github.com/Dudzian/Dudzian/security/code-scanning/59) | Wyłącznie odmowa API; brak podstaw do przypisania bootstrapowi | Stan, severity, rule ID, lokalizacja, instancja, commit i analiza: nieznane |
| [#60](https://github.com/Dudzian/Dudzian/security/code-scanning/60) | Historyczny [komentarz PR #3241](https://github.com/Dudzian/Dudzian/pull/3241#discussion_r4445839602): CodeQL „Clear-text storage of sensitive information”, zapis `initdb.password`, ówczesne `provision.py:365` | Stan, severity, rule ID, aktualna lokalizacja, instancja, commit i analiza: nieznane |
| [#61](https://github.com/Dudzian/Dudzian/security/code-scanning/61) | Historyczny [komentarz PR #3241](https://github.com/Dudzian/Dudzian/pull/3241#discussion_r4445839634): ta sama nazwa reguły, zapis `pgpass`, ówczesne `provision.py:367` | Stan, severity, rule ID, aktualna lokalizacja, instancja, commit i analiza: nieznane |

Komentarze review nie zastępują alertów, wszystkich stron `instances` ani
SARIF. Ewentualna przyszła ocena tych alertów wymagałaby powiązania analizy
z jej metadanymi i SHA; nie jest częścią bieżącej poprawki libpq.
Sam sukces joba CodeQL nie potwierdza naprawienia ani zamknięcia tych alertów.

## Historia źródeł i aktualizacja bazy

Początkową bazą był `06f75ba9c966df876713cc4ef27fee1197b64d99`, po #3241.
`git blame`, porównanie z rodzicem i `original_commit_id` komentarzy wskazują
`a906318f9007ce8f2f2c9c0166cee57bf860b328` jako historyczne wprowadzenie obu
zapisów. To **commit historii kodu**, nie potwierdzony commit alertu.
Nie przypisujemy #59 reguły ani przyczyny na podstawie sąsiednich numerów.

Main `996f2bab4b7ed586a37c0434d857b7ef3a8211b5` zawiera scalone
[#3243](https://github.com/Dudzian/Dudzian/pull/3243) i
[#3245](https://github.com/Dudzian/Dudzian/pull/3245). Wykonana synchronizacja
`4761c43b94d3d2a569e236e10b8ded797913c101` ma rodziców `6ed51759…` i `996f2bab…`.
Siedem plików poprawek natywnych jest identycznych z `ff967ad4…`, a oba pliki
poprawki Linux z `cd3b964a…`. Zachowano oryginalne uchwyty procesu/katalogu,
odmowę nadpisania, precyzyjne zegary, dokładny typ `bytes` i bramki Stage 9/10.
Historyczne błędy brakującego `QueryFullProcessImageName`, rename Errno 87,
prefiksów symlinków i Linux `no-any-return` pochodziły ze starej bazy;
nie są regresjami poprawki poświadczeń. Na nowej bazie oba kontrakty
Windows/Linux przeszły; aktualne bramki produkcyjne opisano osobno poniżej.

## Ograniczenie ryzyka w źródle

[`provision.py`](../../deployment/windows_installer/provision.py) zachowuje
dwa natywne wejścia jawnego tekstu:

1. `secrets.token_urlsafe(48)` → `initdb.password` → `initdb --pwfile=…` →
   początkowe poświadczenie administratora SCRAM.
2. To samo hasło → rekord `pgpass` → jawnie wskazany `passfile` libpq →
   połączenia provisioning i qualification przez psycopg.

`_require_bootstrap_system` sprawdza efektywny token wątku, używając tokenu
procesu wyłącznie po `ERROR_NO_TOKEN`. Inny SID jest odrzucany przed zasobami.
`CreateDirectory` i `CreateFile` otrzymują końcowy deskryptor już przy tworzeniu:
owner SYSTEM, `SE_DACL_PROTECTED`, dokładnie jeden ACE SYSTEM `FILE_ALL_ACCESS`,
brak obcych dziedziczonych ACE oraz `bInheritHandle=False`. Pliki używają
`CREATE_NEW` i share mode 0. Kwalifikacja tego samego uchwytu poprzedza zapis;
odmowa ACL/reparse lub niepełny `WriteFile` przerywa operację. Kwalifikacja
katalogu żąda `READ_CONTROL | ntsecuritycon.FILE_READ_ATTRIBUTES`, zgodnie
z eksportami przypiętego pywin32 312.

Po udanym `initdb` pierwszy plik jest usuwany **przed listenerem**. `pgpass`
pozostaje dłużej, do zakończenia połączeń SQL i końcowego HBA. Hasło nie trafia
do argumentów procesu ani DSN i nie jest dodawane do środowiska potomków.
Wyjście bootstrapowych `initdb`/`pg_ctl` jest odrzucane; diagnostyka eksportuje
tylko walidowane pola i stałe kategorie, bez tekstu wyjątków i łańcucha błędów.

Osobno potwierdzono przyczynę awarii MSI `POSTGRES_CREATE_DATABASE`:
odziedziczone `PGPASSWORD` wygrywało z jawnym `passfile`. `password=""` w DSN
wyłącza to domyślne źródło i pozwala libpq odczytać chroniony plik. Natywny
[test odtwarzający](https://github.com/Dudzian/Dudzian/actions/runs/41620138236)
odrzucił zastane hasło, a identyczne połączenie z passfile i poprawnym hasłem
w pamięci działało. Nie zmieniono SCRAM, TLS `verify-full`, certyfikatów,
końcowego HBA ani bramek kwalifikacyjnych.

Kształt `(PySID, int)` wyniku `TokenUser` potwierdza
[pywin32 b312](https://github.com/mhammond/pywin32/blob/b312/win32/src/win32security.i#L1718).
Przypięty PostgreSQL 17.11 w
[`restricted_token.c`](https://github.com/postgres/postgres/blob/REL_17_11/src/common/restricted_token.c)
ogranicza przywileje/grupy potomka, zachowując SID użytkownika SYSTEM.
Te źródła uzasadniają granice; same nie zastępują testu Windows.

## Oddzielna ocena pozostałego ryzyka

| Alert | Ocena dostępnego historycznego przepływu; bez rozstrzygnięcia statusu API |
| --- | --- |
| #59 | Niezidentyfikowany. Brak podstaw do przypisania, poprawki lub uznania false positive. |
| #60 | `initdb.password` nadal zawiera jawne hasło na dysku. Końcowa DACL obowiązuje od utworzenia, a usunięcie po `initdb` skraca czas życia; nie usuwa klasy plaintext storage. |
| #61 | `pgpass` nadal zawiera jawne hasło do końca SQL. DACL od utworzenia ogranicza czytelników, lecz dłuższy czas życia i możliwe pozostałości po awarii pozostają. |

`FILE_ATTRIBUTE_TEMPORARY` jest wskazówką cache, nie szyfrowaniem ani gwarancją
RAM-only. Wymuszone zabicie procesu może pominąć `finally`, pozostawiając
prywatne pliki i bootstrap postmaster. `unlink` nie gwarantuje wymazania;
Python nie gwarantuje zerowania hasła w pamięci. SYSTEM, administrator
przejmujący własność, dump, pagefile i backup pozostają poza izolacją od
zwykłego czytelnika. Nie dodano automatycznego odzyskiwania po awarii.

Natywny `WriteFile` może nie być modelowany przez CodeQL tak jak wcześniejszy
sink `write_text`. Zniknięcie wyniku nie dowodzi usunięcia plaintext ani ryzyka.
Nie dodano suppressions, nie osłabiono skanerów i nie zmieniano alertów.
`trust`, hasło w argumentach i nowe `PGPASSWORD` odrzucono. DPAPI nie jest
bezpośrednim wejściem `--pwfile`/`passfile`; stdin, named pipe, delete-on-close
lub zamiennik w pamięci wymagają osobnego dowodu zgodności i cyklu życia.

## Historyczna walidacja przed poprawką PGHOSTADDR

Rewizja zweryfikowanego kodu i sondy natywnej:
`3bb4ad27d063000e45886ab3e3bd2d17b87ad715`. Końcowy commit aktualizujący
niniejszy raport nie zmienia kodu; aktualny HEAD gałęzi jest podany w opisie PR.

| Sprawdzenie | Wynik i granica dowodu |
| --- | --- |
| [LocalSystem, run 41709367175](https://github.com/Dudzian/Dudzian/actions/runs/41709367175/job/126173081008) | **PASS**; obie rzeczywiste próby Anonymous/crash wykonane, `CASE_PASSED`, exit0, bez skipów; bootstrap i cały cleanup PASS |
| Modele na tym Windows | **95 passed, 4 skipped**; dwa opt-in ACL/crash uruchomiono dodatkowo pod SYSTEM |
| [Pełny MSI po aktualizacji main](https://github.com/Dudzian/Dudzian/actions/runs/41671594169/job/126082425853) | **PASS**, receipt: install_exit_code=0 i uninstall_exit_code=0; files/services/DACL/PostgreSQL/Production Trust/fail-closed enrollment/cleanup PASS |
| Bieżący zestaw lokalny Windows/Linux po zmianach fixture | **743 passed, 21 platform/opt-in skipped** |
| Ruff całego baseline i zmienionych modułów; osobny format Windows | **PASS** |
| Skonfigurowany mypy Linux i win32 | **PASS**, po 387 plików; dodatkowy dokładny mypy linux-evidence-contract: 19 plików PASS |
| Dokładny `python -m pre_commit run ruff-format --all-files` | **PASS**, bez zmian po końcowej edycji raportu |
| Betterleaks 1.9.0, pełny czysty eksport śledzonych plików z końcowym raportem | **PASS**, exit0, istniejący baseline, bez nowych wyłączeń |
| [CI](https://github.com/Dudzian/Dudzian/actions/runs/41671593966), [Quality](https://github.com/Dudzian/Dudzian/actions/runs/41671594146), [CodeQL](https://github.com/Dudzian/Dudzian/actions/runs/41671593899) i oba workflow testowe Stage 10 dla e1b749bd | **PASS**; sukces CodeQL nie rozstrzyga numerowanych alertów |

[CodeQL dla 3bb4ad27](https://github.com/Dudzian/Dudzian/actions/runs/41709367193),
Style i oba testowe workflow Stage 10 również zakończyły się sukcesem.
Odczyt o 20:18 UTC: Quality i zwykłe CI tej rewizji nadal trwają; Platform
Deployment ma zielone core wszystkich systemów, oba kontrakty, akceptację Windows
i zgodność macOS, lecz MSI job `126177123155` nadal jest w kolejce. Nie nazywamy
całych tych workflow PASS ani nie przenosimy dowodu MSI na cały nowy HEAD.

Sonda Anonymous nadal wymaga rzeczywistego odczytu znanego pliku kontrolnego,
a następnie `Access denied` dla prywatnego katalogu i obu plików. Dostarcza też
atakującemu wcześniej otwarty uchwyt prywatnego katalogu, sprawdzając DACL
samych plików. Odczyty są względne do przypiętych uchwytów fixture, aby prywatne
hostowe katalogi nadrzędne nie maskowały badanej granicy. Udostępnienia,
status pending, brak uchwytu lub inny błąd nie zastępują odmowy dostępu.
Pythonowy kodek UTF-16LE jest inicjalizowany przed impersonacją; osobna regresja
zimnego cache odtwarza wcześniejszy błąd sondy. Nie rozszerzano ACL produkcji,
nie pomijano kontroli ani odmowy reparse. Próba crash potwierdza prywatność
pozostałych plików, **nie automatyczne sprzątanie produkcji po śmierci procesu**.

MSI sprawdzono dla head `e1b749bd6ada920b987847871429d0e2397745ea`.
Receipt zapisuje checkout testowego merge GitHub Actions
`38177dcb4f3a6d67806eb6ca1915529eb4d72c2d`; oba mają identyczne drzewo
`ac0bbb61388162b726cd621236464a18bce26b18`.
Produkcyjny provisioner w tamtej rewizji był identyczny z kodem przed obecną
poprawką PGHOSTADDR, SHA-256
`f75f74973fbc8a2ec1090b49fe5fd29e615f0376e92503766efe9669e07fd40f`.
Zweryfikowano także digest pobranego artefaktu. Po tym przebiegu zmieniono
wyłącznie sondę testową/kodek i raport. Obecna korekta hostaddr/GSS zmienia
produkcyjny instalator i wymaga nowego testu MSI.

Workflow platformowy dla `e1b749bd…` nie był w całości zielony.
Zachowane bramki odmawiają:

- Linux: systemd, clean install i reboot/update mają `MISSING_CURRENT_EVIDENCE`
  (`LIVE_LINUX_INTEGRATION`); `LINUX_PRODUCTION_READY=NOT_READY`.
- Windows: `WINDOWS_REBOOT_RECOVERY`, `WINDOWS_LONG_RUNNING_LIFECYCLE`,
  `WINDOWS_UPDATE_RESTART` i `WINDOWS_FAILED_UPDATE_SAFE_BEHAVIOR` pozostają
  niespełnione. Dwa ostatnie dotyczą nierozstrzygniętego kontraktu bezpiecznego
  updatera, nie tylko niewykonanych testów reboot. `WINDOWS_BETA_RELEASE_READY=NOT_READY`.
- Ręczne próby fizyczne/reboot zostały pominięte zgodnie z bramkami; macOS
  deployment compatibility w tym przebiegu anulowano. Nie raportujemy go jako PASS.

Kod bramek i kontrakty są identyczne ze scalonym main. To odrębne wymagania
kwalifikacji, nie powrót naprawionych błędów API lub Linux no-any-return.

Bez pełnego odczytu alertów, instancji i powiązanego SARIF nie deklarujemy
ich naprawienia ani false positive. Historyczne uznanie HTTP 403 za samodzielny
bloker merge zostało zastąpione bieżącą oceną na początku dokumentu.
Rekomendacja merge tej poprawki zależy od przeglądu kodu i zakończonych testów
nowej rewizji, a nie dostępu do numerowanych alertów. Nie scalono PR #3244
i nie zmieniano stanów alertów ani konfiguracji tokenów/uprawnień.
