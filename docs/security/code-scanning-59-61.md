# Alerty Code Scanning #59, #60 i #61: zakres dowodu

Aktualizacja: 2026-11-03. Raport rozdziela historię kodu, ograniczenie ryzyka
i dane rzeczywiście odczytane z GitHub. PR [#3244](https://github.com/Dudzian/Dudzian/pull/3244)
pozostaje **Draft / BLOCKED** z powodu braku dostępu do Code Scanning API.
PR #3244 nie został scalony, nie zmieniono stanu alertów ani nie zaakceptowano
pozostałego ryzyka w imieniu właściciela. Dowody CI nie upoważniają do
instalacji produkcyjnej, enrollmentu ani fizycznej kwalifikacji Stage 9/10.

## Aktualny dostęp do API

Ponowny odczyt ze skonfigurowanym `GITHUB_ALERT_TOKEN` 2026-11-03:

| Zakres | Czas UTC | Wynik |
| --- | --- | --- |
| `GET /repos/Dudzian/Dudzian/code-scanning/alerts/59`, `/60`, `/61` | 15:29:27–15:29:28 | Wszystkie HTTP 403: `Resource not accessible by integration` |
| Trzy odpowiadające endpointy `/alerts/{number}/instances` | 15:27 | Wszystkie HTTP 403: `Resource not accessible by integration` |
| `GET /repos/Dudzian/Dudzian/code-scanning/analyses` | 15:27 | HTTP 403: `Resource not accessible by integration` |

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
SARIF. Po przywróceniu odczytu trzeba powiązać `analysis_key`, `category`,
`ref` i SHA z analizą oraz pobrać jej SARIF przez
`GET /code-scanning/analyses/{analysis_id}` z `Accept: application/sarif+json`.
Sam sukces joba CodeQL nie potwierdza naprawienia ani zamknięcia tych alertów.

## Historia źródeł i aktualizacja bazy

Początkową bazą był `06f75ba9c966df876713cc4ef27fee1197b64d99`, po #3241.
`git blame`, porównanie z rodzicem i `original_commit_id` komentarzy wskazują
`a906318f9007ce8f2f2c9c0166cee57bf860b328` jako historyczne wprowadzenie obu
zapisów. To **commit historii kodu**, nie potwierdzony commit alertu.
Nie przypisujemy #59 reguły ani przyczyny na podstawie sąsiednich numerów.

Main `996f2bab4b7ed586a37c0434d857b7ef3a8211b5` zawiera scalone
[#3243](https://github.com/Dudzian/Dudzian/pull/3243) i
[#3245](https://github.com/Dudzian/Dudzian/pull/3245). Przygotowana synchronizacja
`4761c43b94d3d2a569e236e10b8ded797913c101` ma rodziców `6ed51759…` i `996f2bab…`.
Siedem plików poprawek natywnych jest identycznych z `ff967ad4…`, a oba pliki
poprawki Linux z `cd3b964a…`. Zachowano oryginalne uchwyty procesu/katalogu,
odmowę nadpisania, precyzyjne zegary, dokładny typ `bytes` i bramki Stage 9/10.
Historyczne błędy brakującego `QueryFullProcessImageName`, rename Errno 87,
prefiksów symlinków i Linux `no-any-return` pochodziły ze starej bazy;
nie są regresjami poprawki poświadczeń. Wyniki nowej rewizji wymagają własnej CI.

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

## Walidacja i warunki dalszej oceny

Natywny bootstrap na `6ed51759…` przeszedł
[run 41660182573](https://github.com/Dudzian/Dudzian/actions/runs/41660182573):
rzeczywisty LocalSystem, przypięte binaria, passfile, SCRAM/`verify-full`,
negatywne hasło/CA/TLS, końcowe HBA i cleanup; modele Windows: 76 passed,
4 opt-in skipped. Nie jest to dowód osobnych prób obcego tokenu i crash.
Lokalne **724 passed, 21 skipped** dotyczą stanu **przed rozszerzeniem** harnessu.

Nowy harness ma uruchamiać istniejące dwa opt-in przypadki LocalSystem:
anonimowy czytelnik i pozostałości po crash. Ich wykonanie na nowej rewizji
pozostaje **oczekujące**, dopóki nie zostanie odczytany wynik Windows.
Modele Win32 i Linux/POSIX nie zastępują tych dowodów Object Managera.

Dokładne SHA, końcowe wyniki pytest/Ruff/mypy i obowiązkowego
`python -m pre_commit run ruff-format --all-files` są utrzymywane w opisie PR.
Bez pełnego odczytu alertów, instancji i powiązanego SARIF nie deklarujemy
ich naprawienia, false positive ani zaakceptowania ryzyka. **Draft / BLOCKED**
pozostaje jawny; żaden PR nie został scalony w ramach tej operacji.
