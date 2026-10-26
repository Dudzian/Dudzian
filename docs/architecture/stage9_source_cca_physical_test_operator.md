# Stage 9 — fizyczna instalacja Source/CCA, TEST ONLY

Stan pozostaje `Stage 9 = IN_PROGRESS / PRODUCTION_NO_GO`. Operator
`scripts/qualify_stage9_source_cca_installation.py` przygotowuje jednorazowy,
syntetyczny odpowiednik `root_source`, `history_source` i `cca` do przyszłej
próby na fizycznym Dell Latitude 5490 z Ubuntu 24.04. Testy repozytorium nie są
wynikiem fizycznym. Status gotowości fizycznej pozostaje
`PHYSICAL_TEST_PENDING`. Nie należy deklarować fizycznego PASS przed
uruchomieniem przez operatora na Dellu i niezależnym sprawdzeniem cleanup.

Pakiet nie zawiera kluczy, custody, dostępu do PostgreSQL ani poświadczeń.
Programy testowe wyłącznie utrzymują proces i obsługują jego zakończenie.
Każda rola otrzymuje odrębny plik wykonywalny Linux, UID/GID, jednostkę systemd
oraz przestrzenie nazw mount, IPC i network. Wspólny interpreter skryptów nie
spełniłby porównania `/proc/PID/exe`.

## Audyt i wykorzystane mechanizmy

Punktem wyjścia jest merge PR #3207,
`8a7cecaab54a00157cd939a1569912bf7166691d`. Przed implementacją przejrzano:

| Źródło | Zastosowanie i ograniczenie |
| --- | --- |
| [Kontrakt instalacji](../../deployment/stage9_trusted_installation_contract.py) | Zamknięty schemat trzech ról, odrębność tożsamości i ścieżek, jawne UID/GID, cgroup i hashe, brak dodatkowych grup. |
| [Chronione pliki](../../deployment/stage9_trusted_installation_files.py) | Odczyty przez zachowane deskryptory, kontrola właściciela, trybu, ACL, symlinków i hardlinków oraz ponowne sprawdzenie obiektów. |
| [Obserwacja runtime](../../deployment/stage9_trusted_installation_runtime.py) | Rzeczywisty systemd, NSS, `/proc`, PID/starttime, executable i cgroup; ograniczone czasowo wywołania istniejących obserwatorów. |
| [Trusted Installation Gate](../../deployment/stage9_trusted_installation_identity_gate.py) | Wiązanie dokładnych bajtów manifestu z osobnym pinem, dwie obserwacje i odrębność namespace. Nawet `MATCHED` zachowuje produkcyjne blokady. |
| [Operator Stage 41B](../../scripts/qualify_stage41b_signer_isolation.py) | Sprawdzone zasady ograniczonych poleceń systemowych, odmowy kolizji, pochodzenia zasobów, obsługi sygnałów i nierekurencyjnego cleanup. Nie jest powielany protokół signerów ani custody. |
| [Przygotowanie Stage 41B](../../scripts/prepare_stage41b_environment.py) | Rozdzielenie nieuprzywilejowanego przygotowania od jawnego wykonania oraz weryfikacja zamkniętego inventory przed operacjami administracyjnymi. Ten test nie instaluje środowiska signerów. |

Codebase Memory MCP nie był dostępny w katalogu narzędzi tej sesji; wnioski
zweryfikowano bezpośrednio w źródłach i odpowiadających testach. Dodatkowy
[opis bramki](stage9_trusted_installation_identity_gate.md) zachowuje znaczenie
wszystkich jej blokad. CLI bramki zwraca udokumentowany kod **2**, także przy
pełnym `installation_checks=MATCHED` i `checks=MATCHED` dla trzech ról.

## Tryby operatora i granica autoryzacji

`--check` tylko odczytuje stan: system i jądro, PID 1, możliwości izolacji,
wymagane programy, konta, jednostki, inne instalacje CryptoHunter oraz
pozostałości wybranego wykonania. Nie naprawia hosta ani nie rezerwuje zasobów.
Obserwacja procesów jest domyślnie ograniczona do dowodów dostępnych bez root:
PID/starttime, wszystkich pól UID/GID, dodatkowych grup i kanonicznego cgroup.
Jednoznacznie niepowiązany proces systemowy nie wymaga odczytu chronionego
`/proc/PID/exe`. Brak dostępu do istotnych metadanych albo ich zmiana daje
`UNCERTAIN`, a nie pozytywny wynik. Gotowość tej kontroli dotyczy wyłącznie
jej jawnego zakresu i nie zastępuje pełnego preflight przed wykonaniem.

`--prepare` tworzy lokalny pakiet bez root. Pakiet zawiera minimalne programy,
jednostki, zamknięty manifest, osobny syntetyczny pin, SHA-256 inventory,
oczekiwane zasoby oraz instrukcje. Należy przygotować go lokalnie na zgodnym
Ubuntu, przejrzeć i zachować jego digest przed wykonaniem uprzywilejowanym.
Nie pobierać skryptów wykonawczych z sieci podczas fizycznej kwalifikacji.

Manifest zasobów `resources.json` wyznacza nazwy testowe:
`chs9t_<run-id>_{r,h,c}`, jednostki `ch-s9t-<run-id>-{r,h,c}.service`
w `/run/systemd/system` i pliki w `/run/ch-s9-test-<run-id>`.
Nie są to nazwy produkcyjnych ról. `inventory.json` wiąże SHA-256 całej
zamkniętej zawartości, w tym `manifest.json`, `TEST_ONLY-pin.json`, jednostek,
programów i instrukcji.

`--execute` wymaga root, dokładnego identyfikatora autoryzacji TEST ONLY,
wskazania fizycznego hosta i oczekiwanego SHA-256 pakietu. Ponawia pełny,
administracyjny preflight i sprawdza pakiet przed zmianami. Kontrola procesów
zachowuje pełny odczyt executable, również dla procesów wykluczonych z zakresu
UID/GID/cgroup zwykłego konta. Ten tryb jest przeznaczony wyłącznie do
świadomego uruchomienia przez operatora na Dellu; nie uruchamiać go w Codexie,
kontenerze ani GitHub Actions. Nie wykonuje automatycznych napraw, nie przejmuje
kont i jednostek oraz nie może przyznać produkcyjnego GO.

`--inspect-cleanup` tylko sprawdza pozostałości konkretnego identyfikatora
wykonania w zakresie dostępnym bez root. Należy go uruchomić osobno po próbie,
także gdy poprzedni raport deklaruje zakończenie cleanup. `scope_clear=true`
oznacza brak wykrytych pozostałości w tym zakresie; `clean=false` i
`full_confirmation=false` nadal wymagają osobnej kontroli administracyjnej.
Kod `0` oznacza powodzenie obserwacji ograniczonej, a nie pełny CLEAN.

`--inspect-cleanup-admin` jest osobnym, jawnym i wyłącznie odczytowym trybem
wymagającym root. Dopiero pełna inspekcja, obejmująca executable oraz wszystkie
dotychczasowe kolizje kont, jednostek, cgroup i ścieżek, może potwierdzić
`clean=true`. Żaden tryb nie przechodzi samoczynnie do sudo ani nie zmienia
zakresu kontroli na podstawie samego UID wywołującego. Sam sukces polecenia
usuwającego nie dowodzi braku zasobów.

Na typowym Ubuntu 24.04 zwykłe konto może odczytać status, stat i cgroup
niepowiązanych procesów, mimo odmowy przy executable roota. Taka odmowa nie
blokuje zakresowego preflight. Ukrywanie procesów przez `hidepid`, nieczytelne
metadane potencjalnego procesu testowego, niekanoniczny cgroup lub niespójny
PID/starttime pozostają przyczyną `UNCERTAIN` i odmowy. Ograniczona obserwacja
nie dowodzi nieobecności dowolnego executable testowego uruchomionego pod inną
tożsamością; tę granicę sprawdza wyłącznie kontrola administracyjna.

## Zaufanie do kodu przed uruchomieniem jako root

Przed uruchomieniem interpretera z UID 0 administrator musi niezależnie
przygotować i zewnętrznie zweryfikować chroniony checkout dokładnego,
uprzednio przejrzanego commitu. Weryfikacja musi obejmować zawartość i ochronę
plików kodu, ich katalogów oraz środowiska importów, w tym modułów ładowanych
dopiero podczas wykonania. Zgodny digest zawartości nie zastępuje kontroli
właściciela, uprawnień i powiązań ścieżek; zmiana trybu pliku bez zmiany jego
bajtów także może unieważnić ochronę. Ten etap jest osobną czynnością
administratora z użyciem uprzednio zaufanych narzędzi, przed wywołaniem
operatora jako root. Pakiet TEST ONLY ani jego syntetyczny pin nie wykonują
tej weryfikacji.

Import Pythona wykonuje kod przed ewentualnym sprawdzeniem go przez ten sam
proces. Dotyczy to także inicjalizacji pakietów, istniejącego bytecode,
plików `.pth`, modułów `sitecustomize`/`usercustomize` i mechanizmów importu.
Flaga `-B` wyłącza zapisywanie bytecode, lecz nie zakazuje odczytu istniejących
plików `.pyc`. Dlatego interpreter, biblioteki, ścieżki wyszukiwania i kod
startowy muszą już pochodzić z niezależnie przygotowanego, chronionego
środowiska. Nie uruchamiać root z kopii należącej do zwykłego użytkownika
w celu samodzielnego potwierdzenia jej zaufania.

Operator kontroluje całe drzewa checkoutu i plikowych lokalizacji wyszukiwania
importów, także ścieżki pakietów, zamiast ograniczać spis do `sys.modules`.
Obejmuje to moduły jeszcze niezaimportowane, inicjalizatory, bytecode i `.pth`.
Wymagany plik `controls` musi istnieć i być chroniony przed jego importem;
brakująca lokalizacja wyszukiwania wymaga ochrony istniejących przodków.
Znany import `controls` odbywa się przed utworzeniem zasobów, po czym kod
ponownie sprawdza ochronę i oczekiwaną lokalizację modułu. Kontrole wracają
przed utworzeniem usług oraz użyciem kontroli odmów. Każde sprawdzenie ponownie
odczytuje właściciela i uprawnienia; zgodne bajty nie pomijają kontroli trybu.

Kontrole wykonywane wewnątrz operatora sprawdzają lokalną zgodność po
zaufanym starcie. Kontrola po imporcie nie cofa wykonania nieufnego modułu
ani nie dowodzi, że aktualnie wykonywane obiekty Pythona odpowiadają bajtom
pliku. Sam proces nie może ustanowić wiarygodności własnego interpretera,
skompromitowanego root/hosta ani zapewnić atomowej ochrony przed zmianami
poza oknem obserwacji. Niezależna weryfikacja przed uruchomieniem pozostaje
obowiązkowa; żaden wynik tych kontroli nie jest niezależną atestacją hosta
lub zgodą produkcyjną.

## Procedura przyszłego uruchomienia na Dellu

Najpierw przejrzeć i pobrać zatwierdzony commit PR zwykłym kontem. Wszystkie
polecenia `-m` uruchamiać z katalogu głównego tego czystego checkoutu.
Flaga `-B` zapobiega tworzeniu plików bytecode podczas kontroli. Nie wykonywać
poniższych poleceń automatycznie. Używać hosta przeznaczonego do izolowanej
próby, bez produkcyjnego custody, baz i poświadczeń. Identyfikator ma 12 małych
znaków szesnastkowych; dla każdej próby wybrać nowy. Przykład `123456abcdef`
poniżej jest wyłącznie przykładem.

Pakiety i raporty trzymać poza checkoutem, aby przygotowanie nie unieważniło
wymagania czystego, zatwierdzonego commitu. Utworzyć osobny katalog zwykłym
kontem i uzyskać jego kanoniczną ścieżkę absolutną:

```sh
mkdir -p ../work
operator_work="$(realpath ../work)"
```

1. Odczytowy preflight zwykłym kontem:

   ```sh
   python3 -B -m scripts.qualify_stage9_source_cca_installation \
     --check --run-id 123456abcdef --uid-base 62000
   ```

   W razie blokady lub `UNCERTAIN` zatrzymać procedurę. Wybrać inny wolny
   zakres UID/GID lub zbadać przyczynę; nie usuwać istniejących kont ani usług
   dla wymuszenia pozytywnego preflight. Brak dostępu do executable
   jednoznacznie niepowiązanego procesu roota sam w sobie nie jest blokadą.

2. Przygotowanie pakietu zwykłym kontem:

   ```sh
   python3 -B -m scripts.qualify_stage9_source_cca_installation \
     --prepare --package "$operator_work/source-cca-package" \
     --run-id 123456abcdef --uid-base 62000
   ```

   Zachować raport przygotowania i podany digest `inventory.json` pakietu.
   Można porównać go z odczytowym
   `sha256sum "$operator_work/source-cca-package/inventory.json"`.
   Przejrzeć inventory,
   role, nazwy, ścieżki oraz zabezpieczenia jednostek. Oczekiwany hash należy
   przenieść do polecenia execute jako literalny, uprzednio sprawdzony SHA-256.
   Nie wyliczać go na nowo z potencjalnie zmienionego pakietu bez ponownego
   przeglądu.

3. Po przygotowaniu pakietu administrator musi osobno umieścić dokładnie ten
   sam przejrzany commit w chronionym checkoutcie, np. `/opt/ch-s9-operator`.
   Samo podanie hasha pakietu nie zabezpiecza importowanego kodu operatora.
   Checkout, importowane pliki i ich przodkowie muszą należeć do root,
   bez zapisu dla grupy lub innych użytkowników i bez niebezpiecznych linków.
   Używać chronionego systemowego `/usr/bin/python3`; nie uruchamiać root
   z interpretera ani modułów należących do zwykłego użytkownika. Przejrzeć
   stan tej kopii i potwierdzić identyczny commit oraz ochronę całego środowiska
   zgodnie z opisaną wyżej granicą zaufania, zanim interpreter zostanie
   uruchomiony jako root. Nie polegać na kontroli operatora wykonywanej dopiero
   po jego zaimportowaniu. Przygotowanie i zewnętrzna weryfikacja chronionej
   kopii są osobnymi czynnościami administratora; `--prepare` jej nie instaluje
   i nie podnosi uprawnień.

   Wyłącznie po świadomej decyzji operatora, na fizycznym Dellu, przejść
   do chronionego checkoutu i uruchomić:

   ```sh
   cd /opt/ch-s9-operator
   sudo /usr/bin/python3 -B -m scripts.qualify_stage9_source_cca_installation \
     --execute --package "$operator_work/source-cca-package" \
     --run-id 123456abcdef --uid-base 62000 \
     --package-sha256 SPRAWDZONY_SHA256_PAKIETU \
     --authorize-test-only 123456abcdef \
     --physical-host Dell-Latitude-5490 \
     --report "$operator_work/source-cca-execution.json"
   ```

   Najpierw przejrzeć kod uruchamiany przez root i jego źródło. Pakiet i raport
   muszą być wskazane jawnymi ścieżkami. Operator weryfikuje dokładny commit;
   syntetyczny pin nie zastępuje tego przeglądu. Kod **2** wykonania pozostaje
   oczekiwanym brakiem kwalifikacji produkcyjnej; wynik próby odczytuje się z
   raportu, łącznie z odmowami i cleanup.

4. Osobna, odczytowa inspekcja pozostałości:

   ```sh
   python3 -B -m scripts.qualify_stage9_source_cca_installation \
     --inspect-cleanup --run-id 123456abcdef --uid-base 62000 \
     --report "$operator_work/source-cca-cleanup-inspection.json"
   ```

   Sprawdzić `scope_clear`, zakres obserwacji i niepewności. Ta komenda nie
   raportuje pełnego CLEAN: nawet przy `scope_clear=true` zachowuje
   `clean=false`, `full_confirmation=false` oraz wymóg inspekcji
   administracyjnej. Jeśli zwykłe konto nie może odczytać wymaganych metadanych
   istotnego procesu, wynik pozostaje `UNCERTAIN` i `BLOCKED`; brak uprawnień
   nie jest dowodem braku zasobu.

5. Administrator może osobno wybrać pełną, nadal wyłącznie odczytową inspekcję:

   ```sh
   cd /opt/ch-s9-operator
   sudo /usr/bin/python3 -B -m scripts.qualify_stage9_source_cca_installation \
     --inspect-cleanup-admin --run-id 123456abcdef --uid-base 62000 \
     --report "$operator_work/source-cca-cleanup-admin.json"
   ```

   Operator nie wykonuje tej komendy automatycznie. Nie tworzy ona zasobów,
   nie usuwa pozostałości i nie uruchamia kwalifikacji. Dopiero kompletny
   raport tego trybu może potwierdzić pełny cleanup. Zachować raport wykonania,
   obie inspekcje i ich SHA-256 razem z commitem i pakietem.

Ewentualny kod `0` bezpiecznego przygotowania, preflight lub inspekcji dotyczy
wyłącznie powodzenia danej operacji. Nie oznacza kwalifikacji hosta, produkcji
ani realizacji Source/CCA.

## Wynik, odmowy i sprzątanie

Po uruchomieniu trzech procesów wykonywana jest rzeczywista implementacja
`inspect_installation()` z PR #3207. Pozytywne porównanie wymaga
`installation_checks=MATCHED` oraz `checks=MATCHED` dla wszystkich trzech ról.
Jednocześnie raport zawsze zachowuje:

```text
TEST_ONLY
status=BLOCKED
qualification=NOT_QUALIFIED
production_emission=BLOCKED
Stage 9 = IN_PROGRESS / PRODUCTION_NO_GO
PHYSICAL_TEST_PENDING
```

Syntetyczne `installation_id` i `approval_id` oznaczają testowy manifest oraz
pin. SHA-256 wiąże dokładne bajty; lokalny pin nie jest niezależnym zatwierdzeniem
produkcyjnym ani atestacją hosta.

Kontrole odmów dotyczą wyłącznie własnych zasobów i obejmują executable,
digest, niedozwolony drop-in, UID/GID, dodatkowe grupy, symlink, uprawnienia,
manifest/pin, niedostępny proces, PID/starttime, cgroup, współdzielone namespace
i niepełną obserwację. Każda odmowa wymaga dodatniej kontroli dostępności i
poprawnej konfiguracji. Raport rozróżnia obserwacje fizyczne i kontrolowane
zmiany wejść obserwacji; symulacja nie dowodzi zmiany konfiguracji jądra.

| Pole `evidence` | Faktycznie wykonywany bodziec |
| --- | --- |
| `PHYSICAL_FILE_FAULT_TEST` | Podmiana executable, symlink lub niebezpieczne uprawnienia własnego pliku. |
| `PHYSICAL_SYSTEMD_FAULT_TEST` | Niedozwolony drop-in własnej jednostki. |
| `PHYSICAL_PROCESS_FAULT_TEST` | Kontrolowane zatrzymanie własnej usługi i ponowne uruchomienie. |
| `FAULTED_TEST_EXPECTATION` | Zmieniony testowy digest, oczekiwane UID/GID, dodatkowa grupa, pin binding lub oczekiwana cgroup; rzeczywisty obserwator pozostaje w użyciu. |
| `OBSERVATION_FAULT_TEST` | Zmieniony snapshot PID/starttime, wspólny namespace lub niepełna obserwacja, po udanej rzeczywistej obserwacji. Nie zmienia fizycznego PID ani namespace. |

Łącznie są to pięć bodźców fizycznych, pięć zmian testowych oczekiwań i trzy
zmiany snapshotu. Wyniki trzech ostatnich nie są fizycznym dowodem odmowy jądra.

Cleanup korzysta z rejestru pochodzenia bieżącego wykonania,
`/run/ch-s9-test-<run-id>/registry.json`. Rejestr na dysku pomaga w inspekcji;
nie upoważnia nowego procesu do przejęcia pozostałości. Nie usuwa zasobów
o niepotwierdzonym pochodzeniu, podmienionych inode, nieznanych plików ani
obcych kont i procesów. Nie używa bezwarunkowego rekurencyjnego usuwania.
Obsługa błędów i `SIGINT`/`SIGTERM`/`SIGHUP` prowadzi przez cleanup; niepowodzenie
cleanup utrzymuje `BLOCKED` i wymaga ręcznego przeglądu.

`SIGKILL`, utrata zasilania lub awaria hosta mogą pozostawić zasoby. Po takim
zdarzeniu uruchomić `--inspect-cleanup` dla zapisanego run ID, zachować raport
i porównać wykazane obiekty z rejestrem i inventory tego wykonania. Nie
uruchamiać ponownie `--execute` dla przejęcia pozostałości. Administrator musi
osobno potwierdzić pochodzenie każdego obiektu przed ręczną interwencją;
nieznane obiekty pozostawić do wyjaśnienia. Po ręcznym cleanup ponownie wykonać
odczytową inspekcję zakresową i osobno pełną inspekcję administracyjną.
Operator nie jest ogólnym narzędziem do kasowania zasobów.

Raport JSON zawiera wersję schematu, dokładny commit, run ID, oznaczenie
syntetyczne, preflight, kontrole dodatnie i odmowy, wynik rzeczywistej bramki,
cleanup oraz SHA-256 dowodów. Nie zawiera sekretów ani pełnych logów hosta.
Niezależne pochodzenie zatwierdzenia, zaufanie do systemd/NSS/procfs,
uwierzytelnienie `SO_PEERPIDFD`, izolacja custody i bibliotek, trwałość
PostgreSQL/CCA, integracja addendum A–C, SIGN/BIND, enrollment, publikacja CCA
i Stage 10 pozostają nierozwiązanymi blokadami produkcyjnymi.

## Walidacja repozytorium i CI

Testy pakietu, lifecycle, odczytowych sond hosta, odmów i CLI bez uprawnień
administracyjnych wchodzą do istniejącego seryjnego `atomic-root-proof-commit`.
Sondy w `tests/security/test_stage9_source_cca_test_probes.py` sprawdzają m.in.
kolizje NSS, obserwacje procfs, programy hosta oraz odmowy bez zmian systemu.
Regresje w `tests/security/test_stage9_source_cca_test_rootless.py` sprawdzają
odmowę dostępu do executable niepowiązanego procesu roota, wykrywanie testowych
UID/GID i cgroup bez tego odczytu oraz niepewność przy niepełnej lub zmienionej
obserwacji. Nie zastępują one fizycznego wykonania na Dellu.
Nowe moduły są objęte mypy Linux/Win32,
Ruff lint/format i połączonym branch coverage atomic/history. Kontrakt
inventory zachowuje wcześniejsze testy, próg changed-code 70%, timeouty oraz
wyłączenia hardware/destructive/network. Żaden workflow nie wywołuje
fizycznego `--execute`. Testy z podstawionymi backendami nie zastępują próby
na Dellu.

## NSS i pozostałości Stage 41B po fizycznym preflight

Odczyt `getent passwd` używa ścisłego UTF-8, także przy ustalonym `LC_ALL=C`.
Międzynarodowy tekst jest dozwolony wyłącznie w polu opisowym GECOS. Pozostałe
pola zachowują ASCII; nazwy kont i członków grup mają jawną składnię, UID/GID
są kanonicznymi dziesiętnymi liczbami ASCII w zakresie 0–4294967294. Parser
sprawdza liczbę pól, bezwzględne ścieżki home/shell, duplikaty nazw i zgodność
wyniku z kluczem zapytania. Wyłącznie LF rozdziela rekordy. Błędne UTF-8,
znaki sterujące i separatory rekordów w opisie powodują odmowę; bajty nie są
pomijane ani zastępowane. Konta o różnych nazwach i wspólnym UID pozostają
widoczne, a niejednoznaczny wynik zapytania o UID blokuje kontrolę.

Samo `/opt/cryptohunter-stage41b` należące do root z trybem 755 oraz brak
aktywnych jednostek **nie ustanawiają bezpiecznej współobecności**. W audycie
źródeł sprawdzono następujące granice:

- `scripts/prepare_stage41b_environment.py` umieszcza instalacje pod pełnym
  commitem w tym prefiksie. Dowodem ukończenia jest `installation.json`
  z trybem 600; ochrona samego prefiksu nie dowodzi ochrony ani integralności
  całego drzewa i nie potwierdza zakończonej instalacji.
- `scripts/qualify_stage41b_signer_isolation.py` używa kont `chs41b*`, ścieżek
  `/run/ch-stage41b-*`, `/run/cryptohunter-stage41b-*` i jednostek o nazwach
  produkcyjnych: `cryptohunter-root-proof-signer.service`,
  `cryptohunter-history-attestation-signer.service` oraz
  `cryptohunter-root-proof-issuer-check.service`. Brak aktywnej jednostki nie
  wyklucza jednostki nieaktywnej, instalacji plikowej ani procesu poza systemd.
- `process_observation(plan)` w Source/CCA bada tożsamości i korzeń wskazanego
  planu Source/CCA. Nie dowodzi nieobecności procesów Stage 41B pod dowolnym
  UID, ani braku otwartych zasobów starego środowiska. Wynik tej obserwacji
  nie może zostać użyty do zatwierdzenia współobecności.

Dlatego `no_other_installations` zachowuje blokadę przy obecnym prefiksie.
Nie dodano wyjątku opartego na nazwie katalogu, właścicielu, trybie 755 ani
braku pasujących aktywnych usług. Ewentualny przyszły wyjątek wymaga osobno
zaprojektowanej pozytywnej weryfikacji izolacji całego starego środowiska;
ten raport jej nie zastępuje.

Zwykły, nieuprzywilejowany `--check` dodaje `preflight.stage41b_diagnostics`:
metadane `lstat` dla `/opt` i prefiksu (w tym typ, symlink, UID/GID i tryb),
nazwy kont i grup `chs41b*`, nazwy jednostek z `list-unit-files` oraz
`list-units --all` i liczby pasujących ścieżek runtime. Spisy są ograniczone
do 128 pozycji; `truncated=true` oznacza niepełny spis. Brak dostępu ma stan
`UNAVAILABLE`, a nie wynik pusty. Raport nie czyta zawartości instalacji,
receiptów ani linii poleceń procesów i nie uruchamia interpretera Stage 41B.
`process_coverage=NOT_ASSESSED_FOR_STAGE41B` jawnie zaznacza brak dowodu
procesowego; `coexistence=BLOCKED` nie jest bramką pozwalającą na wykonanie.

Operator może zebrać raport bez sudo i bez `--execute`, używając dotychczasowego
`--check --run-id <12-hex> --report <nowy-plik.json>`. Nie usuwać katalogów,
kont ani usług na podstawie tej diagnostyki. `PHYSICAL_TEST_PENDING` oraz
`IN_PROGRESS / PRODUCTION_NO_GO` pozostają bez zmian.

Nowa, osobna [diagnostyka współistnienia Stage 41B](stage9_stage41b_coexistence_diagnostic.md)
porównuje niezależnie przypięte inwentarze i rozszerza zakres obserwacji procesów.
Pozostaje wyłącznie diagnostyczna; nie znosi `no_other_installations`.
