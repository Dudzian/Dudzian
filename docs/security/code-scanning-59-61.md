# Alerty Code Scanning #59, #60 i #61: zakres dowodu

Punkt wyjścia analizy: `main` o SHA
`06f75ba9c966df876713cc4ef27fee1197b64d99`, po scaleniu PR #3241.
Analiza dotyczy bootstrapu instalatora Windows; nie upoważnia do instalacji
produkcyjnej, uruchomienia MSI, enrollmentu, rebootu ani merge.

## Dostęp do metadanych GitHub

Dnia 2026-11-02 wywołania `GET /repos/Dudzian/Dudzian/code-scanning/alerts/59`,
`/60` oraz `/61` ze skonfigurowanym `GITHUB_ALERT_TOKEN` zwróciły HTTP 403:
`Resource not accessible by integration`. Nagłówek odpowiedzi dla #59 podaje
`x-accepted-github-permissions: security_events=read`. To brak autoryzacji
Code Scanning, a nie przekroczony limit żądań. Wartości poświadczenia nie
zapisano w logach ani raporcie. Potrzebne jest poświadczenie z odczytem
Code scanning alerts dla tego repozytorium.

Komentarze przeglądu PR są dowodem pomocniczym. Nie zastępują pełnej odpowiedzi
Code Scanning API, listy instancji i SARIF z `codeFlows`. W szczególności nie
potwierdzają aktualnego stanu alertu, severity, rule ID ani kompletnej ścieżki
przepływu raportowanej przez analizator.

| Alert | Ustalenie potwierdzone dostępnym dowodem | Niepotwierdzone |
| --- | --- | --- |
| [#59](https://github.com/Dudzian/Dudzian/security/code-scanning/59) | API zwraca 403; brak przypisania do problemu bootstrapu | Stan, narzędzie, rule ID, severity, lokalizacja, przepływ i commit wprowadzający |
| [#60](https://github.com/Dudzian/Dudzian/security/code-scanning/60) | [Komentarz bota w PR #3241](https://github.com/Dudzian/Dudzian/pull/3241#discussion_r4445839602): CodeQL, „Clear-text storage of sensitive information”, `deployment/windows_installer/provision.py:365` w bazowym HEAD | Aktualny stan, rule ID, severity i ścieżka SARIF |
| [#61](https://github.com/Dudzian/Dudzian/security/code-scanning/61) | [Komentarz bota w PR #3241](https://github.com/Dudzian/Dudzian/pull/3241#discussion_r4445839634): CodeQL, „Clear-text storage of sensitive information”, `deployment/windows_installer/provision.py:367` w bazowym HEAD | Aktualny stan, rule ID, severity i ścieżka SARIF |

W dostępnych komentarzach PR #3241, najnowszej stronie komentarzy review
repozytorium oraz wyszukiwaniu issues nie znaleziono odniesienia do #59.
To ograniczone wyszukiwanie nie dowodzi braku problemu. Nie przypisujemy #59
reguły, pliku ani rozstrzygnięcia na podstawie sąsiednich numerów.

Po uzyskaniu uprawnień należy zachować pełne odpowiedzi dla trzech alertów
i wszystkich stron ich `instances`; powiązać `analysis_key`, `category`,
`ref` i SHA z odpowiednią analizą; pobrać jej SARIF przez
`GET /code-scanning/analyses/{analysis_id}` z
`Accept: application/sarif+json`. Dopiero ten materiał pozwala opisać pełny
przepływ analizatora i rzeczywisty stan GitHub. Sam udany job CodeQL nie
potwierdza zamknięcia określonego alertu.

## Historia i przepływ wynikający ze źródeł

`git blame` obu zapisów, `git show` i porównanie z rodzicem wskazują
`a906318f9007ce8f2f2c9c0166cee57bf860b328` z 2026-11-02 16:54:56 UTC
jako commit wprowadzający oba zapisy hasła. Komentarze bota podają ten sam
`original_commit_id`. Merge `06f75ba9…` wprowadził je do `main`.
To ustalenie historii kodu; bez API nie utożsamiamy go z polem historii alertu.

Przepływy odczytane w bazowym źródle:

1. `secrets.token_urlsafe(48)` → `password` → dodanie końca linii →
   `initdb.password` → ścieżka w `initdb --pwfile=…` → początkowe
   poświadczenie administratora SCRAM.
2. Ten sam `password` → rekord libpq z hostem, portem i użytkownikiem →
   `pgpass` → parametr `passfile` w DSN → kolejne połączenia psycopg/libpq.

Hasło nie jest argumentem `initdb`/`pg_ctl` ani wartością DSN. Argumenty i DSN
zawierają ścieżki do plików. `pg_ctl start/stop` nie wymaga hasła do
administracyjnego połączenia SQL. `initdb` czyta pierwszy plik, libpq drugi;
procesy uruchamiane przez instalator dziedziczą jego tożsamość SYSTEM.

Kształt wyniku `GetTokenInformation(TokenUser)` sprawdzono w
[źródle pywin32 b312](https://github.com/mhammond/pywin32/blob/b312/win32/src/win32security.i#L1718):
jest to para `(PySID, int)`, zgodna z rozpakowaniem użytym przez provisioner.
PostgreSQL 17.11 jest wersją przypiętą w `pins.json` i `contract.py`.
Jego [`restricted_token.c`](https://github.com/postgres/postgres/blob/REL_17_11/src/common/restricted_token.c)
tworzy dziecko z tokenu bieżącego procesu: wyłącza maksymalne przywileje
i grupy Administratorzy/Power Users, przekazuje zero ograniczających SID,
a następnie używa `CreateProcessAsUser`. Nie zmienia SID użytkownika SYSTEM.
To źródłowe uzasadnienie, dlaczego usunięcie grupy Administratorzy nie
odbiera dziecku dostępu przyznanego ACE bezpośrednio SYSTEM. Nie zastępuje
testu odczytu z rzeczywistych przypiętych binariów Windows.

## Cykl życia w bazowym HEAD

Katalog `Security` ma chronione dziedziczne ACL dla Administratorów i SYSTEM
oraz odczyt/wykonanie dla trzech SID usług. `mkdtemp` tworzy pod nim losowy
katalog; dopiero potem `_protect` zastępuje jego DACL dziedziczną polityką
SYSTEM-only, a `_qualify_protected` ją odczytuje i porównuje. Sekret powstaje
dopiero po tym odczycie.

Oba pliki są tworzone przez `Path.write_text` w tym katalogu. Następnie kod
ustanawia i sprawdza ich osobne chronione DACL SYSTEM-only. Dziedziczna
polityka katalogu ogranicza początkowe uprawnienia plików, ale pliki są
otwierane po nazwie, a katalog nie dostaje jawnego deskryptora w operacji
utworzenia. Późniejsze sprawdzenie ACL nie jest samodzielnym dowodem całego
okresu od utworzenia ani zachowania natywnych czytelników Windows.

Pliki pozostają do końca konfiguracji schematu, kwalifikacji i przełączenia
końcowego HBA. `finally` próbuje zatrzymać bootstrap PostgreSQL i usunąć
katalog. Pierwotny wyjątek zachowuje tożsamość; dodatkowa notatka cleanup
zawiera jedynie klasę wyjątku. Obsługa CLI zapisuje allowlistowane pola
operacji/etapu/klasy, bez tekstu wyjątku. Nieobsługiwane zabicie procesu
nie wykonuje Pythonowego `finally`: pliki mogą pozostać na dysku z ostatnią
DACL, a zatrzymanie bootstrap postmastera nie jest wtedy dowiedzione.
Usunięcie pliku nie jest gwarancją wymazania bloków, kopii zapasowych ani
zrzutów pamięci.

## Ocena alternatyw

| Wariant | Ocena bezpieczeństwa i zgodności |
| --- | --- |
| Przywrócenie `trust` lub brak początkowego poświadczenia | Odrzucone: tworzy okno wejścia dowolnego lokalnego klienta przed końcowym HBA. |
| Hasło w argumentach procesu | Odrzucone: ujawnia sekret w widocznej linii poleceń i diagnostyce subprocess. |
| `PGPASSWORD` w środowisku potomków | Odrzucone: rozszerza zasięg sekretu o środowiska procesów i diagnostykę; `pg_ctl` nie potrzebuje sekretu. |
| DPAPI/szyfrowany plik przekazany jako `--pwfile`/`passfile` | Nie jest bezpośrednim zamiennikiem: `initdb` i libpq oczekują jawnego tekstu. Adapter odszyfrowujący musi sam mieć bezpieczny kanał i cykl życia. |
| Bezpośrednie hasło tylko w pamięci psycopg | Może usunąć drugi plik; wymaga sprawdzenia wszystkich wywołań provisioning/qualification, reprezentacji DSN i błędów, aby sekret nie trafił do logów. Nie rozwiązuje wejścia `initdb`. |
| stdin/prompt `initdb` | Wymaga dowodu obsługi nieinteraktywnego Windows przez przypięte binaria; nie należy zakładać, że plik `-` lub konsola jest bezpiecznym stdin. |
| Nazwany potok Windows | Wymaga dowodu zgodności sposobu otwierania przez `initdb`, jawnej DACL potoku, ochrony instancji, synchronizacji i fail-closed przy błędzie lub śmierci procesu. |
| Krótkotrwały plik z jawną DACL przy tworzeniu | Zachowuje natywne wejście PostgreSQL. Musi od początku odrzucać dziedziczenie, istniejący cel/reparse point i obcych czytelników; potrzebuje osobnego dowodu cleanup po awarii procesu i zgodności flag współdzielenia. |

`FILE_FLAG_DELETE_ON_CLOSE` nie może zostać przyjęty bez sprawdzenia
zgodności z otwarciem pliku przez `initdb`: czytelnik musi użyć zgodnych flag
współdzielenia usuwania. Modele ACL oraz test Linux/POSIX nie dowodzą tej
własności natywnego Windows.

## Zastosowane ograniczenie ryzyka

Zachowano dwa krótkotrwałe pliki wymagane przez wybrany natywny interfejs;
nie dodano kanału potokowego ani hasła w DSN. Nowe helpery w
[`provision.py`](../../deployment/windows_installer/provision.py)
ustanawiają następujący kontrakt:

1. `_require_bootstrap_system` odczytuje efektywny token wątku. Token procesu
   jest używany wyłącznie po `ERROR_NO_TOKEN`, nie po dowolnym błędzie.
   SID inny niż LocalSystem jest odrzucany przed utworzeniem zasobów.
2. `_bootstrap_security_attributes` buduje jawny deskryptor: owner SYSTEM,
   ustawione `SE_DACL_PROTECTED`, dokładnie jeden dozwalający ACE SYSTEM
   z `FILE_ALL_ACCESS`. Tylko ACE katalogu ma `OBJECT_INHERIT_ACE` oraz
   `CONTAINER_INHERIT_ACE`; uchwyty mają `bInheritHandle=False`.
3. `CreateDirectory` otrzymuje ten deskryptor przy tworzeniu losowego
   katalogu. Nie ma etapu tworzenia z dziedzicznymi czytelnikami `Security`.
   Odczyt deskryptora z uchwytu sprawdza owner, ochronę DACL, dokładny ACE
   i brak reparse point przed wygenerowaniem hasła.
4. Każdy `CreateFile` dostaje końcowy deskryptor SYSTEM-only, `CREATE_NEW`
   i tryb współdzielenia 0. To natywna operacja utworzenia ustanawia DACL;
   późniejsza kwalifikacja nie służy jej naprawianiu. Odczyt zabezpieczeń
   tego samego uchwytu poprzedza `WriteFile`; błędne ACL i niepełny zapis
   kończą operację błędem. Uchwyt zamyka się również po wyjątku.
5. `initdb.password` jest usuwany natychmiast po udanym `initdb`, przed
   startem listenera. `pgpass` trwa do końca połączeń SQL. Żadne z tych
   narzędzi nie otrzymuje hasła w argumentach ani w nowej zmiennej środowiska.
   `stdout` i `stderr` bootstrapowych `initdb` oraz `pg_ctl` trafiają do
   `DEVNULL`, aby tekst narzędzi nie przeszedł do logów instalatora.
6. Końcowe HBA, mapowania certyfikatów, SCRAM bootstrapu i `verify-full`
   pozostają takie same. Ścieżka w DSN wskazuje wyłącznie prywatny passfile;
   poświadczenie nie jest wpisywane do DSN ani konfiguracji PostgreSQL.
7. CLI obsługuje także błąd publikacji bezpiecznej diagnostyki. Wyjątek
   `write_safe_failure` jest przechwytywany wewnątrz obsługi pierwotnego błędu;
   stderr otrzymuje tylko stałe komunikaty i nazwy klas. Błąd zapisu raportu
   nie powoduje nieobsłużonego tracebacku z łańcuchem zawierającym prywatny
   tekst pierwotnego wyjątku. Operacja nadal zwraca kod błędu.

Jest to ochrona dostępu od momentu utworzenia, **nie szyfrowanie**. Hasło
nadal jest jawnym tekstem w dwóch plikach, choć pierwszy żyje krócej.
`FILE_ATTRIBUTE_TEMPORARY` jest wskazówką dla cache, nie gwarancją pracy
wyłącznie w RAM. Zastąpienie `Path.write_text` natywnym `WriteFile` wynika
z potrzeby atomowego utworzenia z SECURITY_ATTRIBUTES i kwalifikacji
uchwytu. CodeQL może nie modelować tego natywnego sinka; ewentualne
zniknięcie wyniku po tej zmianie nie dowodzi usunięcia zapisu jawnego tekstu
ani naprawienia klasy ryzyka. Nie dodano wyciszenia lokalnego/globalnego
ani maskowania wartości sekretu.

Po zwykłym wyjątku nadal działa cleanup z zachowaniem pierwszej przyczyny.
Po wymuszonym zakończeniu procesu, awarii hosta lub błędzie usuwania mogą
pozostać pliki SYSTEM-only; może też pozostać bootstrap postmaster.
Administrator z uprawnieniami przejęcia własności, SYSTEM, zrzut pamięci,
pagefile lub backup pozostają poza gwarancją izolacji od zwykłego użytkownika.
Nie deklarujemy zerowania obiektów Pythona, bezpiecznego wymazywania nośnika
ani automatycznego usunięcia po śmierci procesu. Dalsze ograniczenie tych
pozostałości wymaga zgodnego i natywnie zweryfikowanego kanału wejścia
`initdb` lub osobnego kontraktu odzyskiwania po awarii.

## Rozstrzygnięcia i granice walidacji

| Alert | Decyzja w tej analizie | Rzeczywisty stan GitHub |
| --- | --- | --- |
| #59 | Nadal otwarty: niezidentyfikowany bez autoryzowanego API; brak podstaw do poprawki lub false positive | Niepotwierdzony — API 403 |
| #60 | Nadal otwarty: wprowadzono ograniczenie dostępu i czasu życia pierwszego pliku, ale jawny zapis pozostaje; brak podstaw do deklaracji naprawienia lub false positive | Niepotwierdzony — API 403 |
| #61 | Nadal otwarty: wprowadzono ACL przy tworzeniu passfile, ale jawny zapis i pozostałości po awarii pozostają; brak podstaw do deklaracji naprawienia lub false positive | Niepotwierdzony — API 403 |

Nie zaakceptowano ryzyka w imieniu właściciela i nie zmieniono stanu żadnego
alertu. Ocena pozostałego ryzyka wymaga decyzji właściciela oraz pełnej
identyfikacji alertów. Testy modelu Win32 na Linux sprawdzają przekazywane
deskryptory, kolejność i ścieżki odmowy; nie dowodzą działania Windows Object
Manager, izolacji obcego tokenu ani dostępu rzeczywistego `initdb`/libpq.
Natywne testy Windows mają osobny opt-in i nie są zastępowane testem
PostgreSQL na Linux. Brak ich wykonania oraz brak metadanych API wymagają
pozostawienia PR jako Draft. Końcowe wyniki poleceń i SHA walidacji należy
czytać w opisie PR; dokument nie przypisuje wyniku niewykonanym testom.
