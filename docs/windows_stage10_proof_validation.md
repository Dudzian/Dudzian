# Windows Stage 10 — audyt i warunki kwalifikacji dowodów

## Zakres i bieżąca decyzja

Audyt dotyczy wyłącznie `WINDOWS_REBOOT_RECOVERY` oraz
`WINDOWS_LONG_RUNNING_LIFECYCLE`. Podstawą są źródła na rewizji
`e5d54cc4a41c008a17f577226e91b5654f94a4aa` i poprawki tego PR. Codebase Memory
MCP i lokalny CLI nie były dostępne; sprawdzono bezpośrednio wskazane pliki,
powiązane kwalifikatory Windows, producenta i odbiorcę evidence oraz testy.
Nie wykonywano restartu, próby 24h ani operacji na fizycznym hoście.

Wersja `cryptohunter.windows.stage10.lifecycle.v1` nie ma zaimplementowanego
autorytetu, który uzasadniałby komplet jej deklaracji. Poprawka usuwa możliwość
wydania `PASS`: każda deklaracja ma jawny stan `BLOCKED`, producent evidence
odmawia publikacji `PASS` z receipt v1, a walidacja downstream odrzuca taki wynik
również po pominięciu producenta. Nazwa dowodu, poprawna struktura JSON,
zgodne identyfikatory CI ani tekst `PASS` nie zastępują obserwacji.

`windows_stage10_live.ps1` jest obecnie punktem odmowy. Kończy się przed
przygotowaniem stanu reboot, odczytem produktu i rozpoczęciem soak. Nie zbiera
częściowych obserwacji. Usunięcie nieuzasadnionych ścieżek `PASS` nie oznacza
implementacji fizycznego kwalifikatora. Implementacja i przegląd brakujących
obserwatorów są warunkiem przed autoryzacją późniejszej próby.

Stage 10 pozostaje `BLOCKED`. Ten PR nie zmienia statusu Stage 9, legal production
enrollment ani bramek produkcyjnych. Nie tworzy nowego systemu Production Trust.

## Tymczasowa rola `require_native_proof_authority()`

Ta funkcja jest świadomą, nieprzechodnią blokadą tymczasową: zawsze zgłasza
`WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE`. Nie zbiera obserwacji,
nie weryfikuje fizycznej próby i nie stanowi ukończonego kwalifikatora.
Zachowanie blokady testuje się niezależnie w producencie i konsumencie evidence,
żeby ręcznie przygotowany pozytywny JSON nie omijał odmowy.

Przyszłe zastąpienie blokady wymaga odrębnego natywnego autorytetu, który połączy
chronioną historię bootu, pochodzenie startu usługi, ciągle utrzymywany uchwyt
procesu, nieprzerwany pomiar monotoniczny oraz bieżącą kwalifikację zatwierdzonego
produktu i SQL przez istniejące autorytety Production Trust. Zapis diagnostyki
nie dostarcza żadnej z tych władz. Bieżąca remediacja nie implementuje tego
autorytetu i nie zmienia warunków jego późniejszego przeglądu ani autoryzacji.

## Mapa deklaracji dowodowych

Tabela obejmuje osiem deklaracji w dwóch proof IDs; powtórzenie
`current_revision_runtime_bound` jest wymagane w obu kontekstach.

| Proof ID / deklaracja | Wymagane obserwacje | Źródło zaufania | Istniejący weryfikator | Brakująca kontrola i bieżący wynik |
| --- | --- | --- | --- | --- |
| `WINDOWS_REBOOT_RECOVERY` / `boot_changed` | Ta sama zatwierdzona maszyna przed i po próbie; dwie rzeczywiste tożsamości bootu; nowszy boot; ciągłość tej samej próby, rewizji i CI run/attempt. | Natywny Windows oraz chroniony kontekst próby pochodzący spoza receipt. | Dotychczasowe `Machine-Guid` i `Boot-Time` w `windows_stage10_live.ps1` porównywały rejestr/WMI z zapisanym stanem. | Brak uwierzytelnionej ciągłości stanu i kompletnego obserwatora bootu. Dowolnie zapisany `boot_before` nie dowodzi restartu. `BLOCKED`. |
| `WINDOWS_REBOOT_RECOVERY` / `current_revision_runtime_bound` | Zatwierdzone MSI, manifest i clean-install receipt o niezależnie przypiętych hashach; ponowna weryfikacja zainstalowanego payloadu, MSI registration, ścieżki i creation time działającego procesu SCM po boot. | Zatwierdzony handoff artefaktu i istniejący Production Trust, rzeczywisty SCM oraz natywny proces. | `windows_stage10_artifact_preflight`, `windows_stage9_host_inspection._qualified_install_inputs`, `qualify_enrolled_runtime(live=True)`; pomocniczo `qualify_runtime_provenance`. | Sam zgodny hash EXE i dwa spójne JSON nie dowodzą zatwierdzonego wydania. Brak pełnego powiązania z obserwatorem Stage 10. `BLOCKED`. |
| `WINDOWS_REBOOT_RECOVERY` / `service_autostarted` | Uruchomienie właściwej instancji przez SCM w wyniku konkretnego bootu; wykluczone późniejsze ręczne uruchomienie, zmiana usługi i luka w obserwacji. | Niezależna natywna obserwacja pochodzenia startu powiązana z bootem i procesem. | `windows_service_recovery.configure_recovery` weryfikuje konfigurację i bounded failure actions; nie dowodzi przyczyny startu. | Brak obserwatora pochodzenia startu. `Automatic` i późniejsze `Running` są niewystarczające. `BLOCKED`. |
| `WINDOWS_REBOOT_RECOVERY` / `readiness_contract_after_reboot` | Świeża gotowość rzeczywistego backendu po boot; zgodny SCM PID, creation time, ścieżka i zatwierdzony payload; ACL; obecna autentykowana obserwacja PostgreSQL i rozdziału ról. | Istniejący kwalifikator enrolled runtime oraz uwierzytelniony kanał do rzeczywistych procesów usług. | `qualify_enrolled_runtime(live=True)`, `_readiness`, `_product_acl`, `windows_stage9_service_evidence.collect_service_postgresql`. | JSON jest markerem startowym, bez świeżego challenge i trwałego potwierdzenia łączności. Brakuje wymaganych autorytetów enrollment/SQL i integracji Stage 10. `BLOCKED`. |
| `WINDOWS_LONG_RUNNING_LIFECYCLE` / `duration_at_least_24h` | Co najmniej 86 400 sekund według odpowiedniego zegara monotonicznego w jednej nieprzerwanej próbie; ustalone zasady suspend/resume; integralność zakresu i związku z obserwowaną instancją. | Monotoniczny zegar i ciągle działający kwalifikator związany z hostem, bootem i próbą. | Poprzedni skrypt używał `System.Diagnostics.Stopwatch`; UTC było tylko opisem. | Sam wpis czasu w receipt nie uwierzytelnia przebiegu. Brak kompletnego obserwatora oraz obsługi przerwania/suspend. `BLOCKED`. |
| `WINDOWS_LONG_RUNNING_LIFECYCLE` / `current_revision_runtime_bound` | Ten sam zatwierdzony produkt w całym zakresie próby; ponowna weryfikacja wejść i payloadu; obraz i tożsamość rzeczywistego procesu, bez podmiany MSI/EXE. | Ten sam zatwierdzony handoff, Production Trust, SCM i natywny proces co dla reboot. | Artifact preflight, `_qualified_install_inputs`, `_installed_files`, `_registration`, `qualify_enrolled_runtime(live=True)`. | Poprzednie okresowe hashowanie pliku nie dowodzi ciągłości załadowanego obrazu ani zatwierdzenia manifestu. `BLOCKED`. |
| `WINDOWS_LONG_RUNNING_LIFECYCLE` / `same_service_instance` | Ciągła obserwacja oryginalnego procesu i jego zakończenia; zgodne SCM PID, creation time, ścieżka i obraz; brak luk lub utraty uchwytu. | Natywny uchwyt do konkretnego procesu i ciągły obserwator życia, powiązany z SCM. | `Assert-Stage10SoakIdentity` wykrywa różnice PID i czasu startu pomiędzy dwiema poprawnymi próbkami; `_stable_service_instances` sprawdza końce krótkiej inspekcji. | Żaden z tych testów porównawczych nie dowodzi ciągłości przez 24h. Próbki co 60 s nie wystarczają, także przy ponownym użyciu PID. `BLOCKED`. |
| `WINDOWS_LONG_RUNNING_LIFECYCLE` / `readiness_contract_before_after` | Dwie świeże, autentykowane obserwacje na końcach zakresu, związane z tą samą stale obserwowaną instancją i zatwierdzonym produktem; wymagane dowody SQL. | Bieżąca kwalifikacja produktu i nieprzenośne autorytety SQL, związane z natywnymi procesami usług. | `qualify_enrolled_runtime(live=True)`, `_readiness`, `collect_service_postgresql`, `summarize_passive_postgresql`. | Ten sam marker JSON na obu końcach nie dowodzi świeżości ani łączności PostgreSQL; brak ciągłego zakresu procesu. `BLOCKED`. |

## `backend-readiness.json`: granice znaczenia

Źródłem markera jest
[`windows_installer/backend_service.py`](../deployment/windows_installer/backend_service.py),
funkcja `_run`. Backend na początku otwiera połączenie PostgreSQL, sprawdza
`session_user/current_user`, zamyka połączenie, wykonuje próby odmowy obcych ról
i odczytu obcego klucza, a następnie buduje CoreHost. Po zgłoszeniu `SERVICE_RUNNING`
zapisuje marker przez plik tymczasowy i `os.replace`. Dalsza pętla nie odświeża
markera ani nie utrzymuje sprawdzanego w ten sposób połączenia PostgreSQL.

Marker zawiera `pid`, `db_role`, `corehost_lock`, `cross_key_denied` i
`cross_roles_denied`. Nie zawiera creation time, boot identity, rewizji, CI
run/attempt, wersji produktu, SHA-256 programu, heartbeat ani challenge.
Atomowa zamiana ogranicza częściowy zapis; nie uwierzytelnia zawartości.

| Zagrożenie | Ocena i wymagana kontrola |
| --- | --- |
| Nieaktualny raport lub odtworzenie starego pliku | Poprzedni Stage 10 nie sprawdzał nawet czasu zapisu. `_readiness` z host inspection sprawdza `written >= process_started`; to kontrola spójności, nie niezależna władza nad treścią. Potrzebna jest świeża kwalifikacja procesu i autentykowane wyzwanie. |
| Inny proces lub ponownie użyty PID | Zgodność samego PID nie wystarcza. Wymagane są creation time, natywna ścieżka obrazu, rzeczywisty proces SCM i zachowana tożsamość procesu podczas weryfikacji. |
| Wersja lub SHA-256 dotyczą pliku innego niż proces | Hash SCM `PathName` i `DisplayVersion` nie wiążą same w sobie załadowanego procesu z zatwierdzonym wydaniem. Istniejący kwalifikator sprawdza cały payload, MSI registration i procesy; wejściowe piny pochodzą z zatwierdzonego handoff. |
| Nieautoryzowana zmiana pliku lub rodzica | Potrzebna jest bieżąca kontrola całego właściwego drzewa ACL, właścicieli i przekierowań, z istniejących kwalifikatorów produktu. Sam poprawny JSON nie potwierdza ochrony. Uprawniony do zapisu backend także nie staje się niezależnym autorytetem całej kwalifikacji. |
| Utrata PostgreSQL po starcie | Marker pozostaje bez zmian, a początkowe połączenie jest już zamknięte. Nie dowodzi aktualnej łączności ani braku przerw. Bieżąca obserwacja SQL musi pochodzić z rzeczywistego procesu usługi; wymagana ciągłość połączenia wymaga osobnego obserwatora. |
| `cross_key_denied` | Backend przyjmuje dowolny `OSError` przy odczycie obcego klucza, także brak pliku. Pole nie rozróżnia przyczyny odmowy. Nie wolno podczas audytu odczytywać kluczy; istniejące `provision.qualify_dacl` i `_product_acl` sprawdzają właściwą ochronę. |
| `cross_roles_denied` | Pole podsumowuje próby wykonane przy starcie. Aktualna macierz wymaga rzeczywistych, ograniczonych czasowo prób pod tożsamościami usług, sprawdzenia wyników/SQLSTATE i związku z konkretną instancją PostgreSQL. Błąd sieci lub timeout nie oznacza odmowy uwierzytelnienia. |

Właściwą istniejącą granicą SQL jest
[`windows_stage9_service_evidence.collect_service_postgresql`](../deployment/windows_stage9_service_evidence.py).
Weryfikuje dwa rzeczywiste procesy usług przez uwierzytelniony lokalny pipe,
świeże nonce i collection ID, ograniczony czas żądania, surowe odpowiedzi SQL,
tożsamość instancji i ponowną kontrolę procesów oraz ACK. Wynikowy autorytet to
lokalny, nieprzenośny receipt. `summarize_passive_postgresql` sprawdza jego rejestr,
proces wystawcy, wiek monotoniczny do 60 s, binding host/process i aktualny
walidator. Serializowany raport pozostaje diagnostyką i nie odzyskuje tej władzy.

Należy używać
[`qualify_enrolled_runtime(live=True)`](../deployment/windows_stage9_install_qualification.py)
z niezależnymi pinami zatwierdzonych wejść, a nie uznawać zapisany raport tej
funkcji za nowy autorytet. Aktualny
[`load_windows_external_provisioning_handoff`](../deployment/windows_installer/corehost_composition.py)
odmawia z `PRODUCTION_TRUST_UNAVAILABLE` albo
`LEGAL_PRODUCTION_ENROLLMENT_NOT_COMPLETED`. Dodatkowo autorytet pełnego
`active_hba` pozostaje `BLOCKED`: parsowane pliki HBA i próbki odmów nie dowodzą
całego aktywnego zestawu reguł. Są to oczekiwane odmowy. Nie wolno ich obejść
przez JSON, deklarację operatora, test fixture ani zmniejszenie wymagań.

## Przerwania i publikacja wyniku

| Zdarzenie | Wymagane zachowanie kwalifikatora |
| --- | --- |
| Nieudany restart lub brak zmiany bootu | Brak `boot_changed=PASS`; zachować odmowę i diagnostykę. |
| Ręczny start po boot | Brak `service_autostarted=PASS`, nawet przy `Automatic` i `Running`. |
| Zmiana PID, creation time, ścieżki lub obrazu | Odrzucić ciągłość danej instancji; nowy proces nie kontynuuje starej próby. |
| Restart pomiędzy próbkami | Bez ciągłego obserwatora wynik pozostaje `BLOCKED`, nawet przy zgodnych końcowych próbkach. |
| Utrata PostgreSQL | Nie wyprowadzać gotowości z zachowanego JSON; wymagane świeże SQL. Brak wymaganego dowodu oznacza odmowę. |
| Podmiana readiness, MSI, EXE lub zatwierdzonych wejść | Odrzucić integralność i powiązanie produktu; nie aktualizować oczekiwanych hashy na podstawie zmienionych danych. |
| Zmiana hosta, runnera, rewizji, CI run ID lub attempt | Odrzucić próbę; historyczny raport z innej tożsamości nie jest kontynuacją. |
| Przerwanie kwalifikatora, utrata uchwytu lub luka obserwacji | Nie odtwarzać ciągłości z czasu UTC ani zapisanego licznika. Nowa próba wymaga własnego pełnego zakresu. |
| Brak obserwacji, nieznany status, błędny typ lub niepełny raport | Nigdy nie domyślać `PASS`; odmowa dotyczy każdej zależnej deklaracji. |
| Błąd zapisu końcowego lub częściowy plik | Kod zakończenia oznacza błąd; brak publikacji evidence. Stary wynik nie może zostać uznany za wynik bieżącej próby. |

Niedokończony plik ani sam sukces wcześniejszego etapu nie mogą zamykać bramki.
Raport odmowy pozostaje diagnostyką nawet po poprawnej publikacji. Producent
nie wystawia pozytywnego receipt, a konsument nadal odrzuca obie deklaracje
lifecycle z `PASS`, niezależnie od pochodzenia i daty dostarczonego JSON.

## Remediacja zapisów diagnostyki po przeglądzie PR #3236

W rewizji `727106629d84f0f91203af44fe19d65276720cba` funkcja zapisująca raport
oraz producent evidence używały `output.unlink(missing_ok=True)` przed dalszą
walidacją. Parametr wyjściowy nie upoważniał do usuwania istniejących danych.
Takie usunięcie mogło naruszyć inny raport lub chroniony artefakt, także gdy
bieżąca próba kończyła się oczekiwaną odmową. Nie było właściwym sposobem
ochrony przed ponownym wykorzystaniem historycznego `PASS`.

Po remediacji domyślnym wyjściem modułu
`deployment.windows_stage10_proof_validation` jest JSON `BLOCKED` na stdout.
Ta ścieżka nie otwiera plików wynikowych. `windows_stage10_live.ps1` zachowuje
parametry legacy `-State` i `-Output` dla zgodności wywołań, lecz nie używa ich
do zapisu diagnostyki. Przy osobnym żądaniu zapisu przekazuje je wyłącznie jako
chronione wejścia, podobnie jak `-QualifiedManifest` i `-CleanInstallReceipt`.
Nie pobiera z nich upoważnienia do zastępowania danych.

Zapis do pliku jest oddzielnym żądaniem:

- Python wymaga `--approved-output-directory` wskazującego bezwzględny,
  istniejący i osobno zatwierdzony katalog. Opcjonalne `--output` musi oznaczać
  nowy plik `.json` bezpośrednio w tym katalogu. Bez `--output` powstaje nowa
  nazwa `stage10-blocked-<attempt_id>.json`. Powtarzalne `--protected-input`
  wskazuje dodatkowe chronione ścieżki, które nie mogą stać się wyjściem,
  także jeżeli taki plik wejściowy jeszcze nie istnieje.
- PowerShell wymaga `-ApprovedDiagnosticDirectory`; opcjonalny
  `-DiagnosticOutput` odpowiada wyłącznie nowemu plikowi w tym katalogu.
  Sam `-DiagnosticOutput` bez zatwierdzonego katalogu powoduje odmowę.
- Writer nie tworzy katalogu, nie naprawia właściciela ani uprawnień, nie
  nadpisuje plików i nie usuwa istniejących nazw. Kolizja z wejściem,
  manifestem, kontraktem, chronionym artefaktem lub starszym raportem powoduje
  odmowę. Nie ma opcji zastąpienia ani zgody wywiedzionej ze starego `PASS`.

Obserwacje i operacje plikowe są związane z natywnymi uchwytami:

| Granica | Egzekwowane warunki |
| --- | --- |
| Wspólna walidacja celu | Bezwzględna ścieżka katalogu, brak przejść `..`, ograniczona nazwa nowego JSON bez przekierowań i nazw urządzeń; bezpośredni potomek zatwierdzonego katalogu i odmowa jawnych aliasów chronionych wejść. |
| Systemy POSIX | Katalog końcowy należy do bieżącego UID i nie pozwala grupie ani innym użytkownikom na zapis. Łańcuch katalogów otwierany jest przez `O_NOFOLLOW`; sprawdzane są właściciel, uprawnienia i tożsamość inode. Kontrolowane katalogi sticky, takie jak `/tmp`, są dopuszczone jako przodkowie chronionego katalogu operatora. Utrzymane deskryptory katalogów wiążą tworzenie i publikację przez `dir_fd`; podmiana ścieżki nie przekierowuje zapisu. |
| Windows | Wymagany jest lokalny NTFS oraz istniejące kwalifikatory ochrony filesystem. Łańcuch natywnych uchwytów odrzuca symlinki, junctions i inne reparse points. Uprawnienia końcowego katalogu i pliku są ograniczone do bieżącego SID, SYSTEM i Administratorów; przodkowie uwzględniają także TrustedInstaller. Tworzenie pliku odbywa się względem zachowanego uchwytu katalogu przez natywne API, a publikacja przez uchwyt pliku, bez ponownego rozwiązywania ścieżki. |
| Publikacja | Nowy plik roboczy `.partial` powstaje wyłącznie przez tworzenie z odmową kolizji. Po pełnym zapisie, synchronizacji i kontroli tożsamości otrzymuje nazwę końcową atomową operacją bez zastępowania: natywny no-replace na POSIX albo handle-relative rename na Windows. Wygrana konkurencyjnej publikacji nie upoważnia do jej nadpisania. |

Model uprawnień ufa właścicielowi zatwierdzonego katalogu i administratorowi
systemu. POSIX wiąże katalog przez deskryptor, ale źródło natywnego rename
pozostaje nazwą własnego pliku roboczego w prywatnym katalogu. Kontrola inode
nie stanowi atomowej ochrony przed wrogim procesem z tym samym UID, który ma
prawo zmieniać ten katalog. Taki proces lub administrator może również zmienić
same uprawnienia. Nie wywodzimy z ochrony filesystem autorytetu fizycznego.

Przy błędzie lub przerwaniu proces zwraca odmowę. Własny `.partial` może pozostać
w zatwierdzonym katalogu do osobnego przeglądu; nie jest usuwany przez ponownie
rozwiązywaną nazwę i nie jest raportem ani dowodem. Błąd wykryty po publikacji
może pozostawić kompletną diagnostykę `BLOCKED`; także jej nie wycofuje się
usuwaniem. Częściowy JSON, kompletny raport odmowy i historyczny plik `PASS`
nie uzyskują władzy evidence przez obecność na dysku.

Każda nowa diagnostyka ma własny `attempt_id`. To znacznik rozróżniający raporty,
nie źródło zaufania ani autoryzacja fizycznej próby. Ochrona przed starym `PASS`
wynika z odmowy producenta i konsumenta, a nie z kasowania starego pliku.
Testy regresyjne korzystają wyłącznie z izolowanych katalogów tymczasowych;
sprawdzają kolizje, ochronę wejść, uprawnienia, przekierowania, podmiany ścieżek,
błędy zapisu i przerwania. Nowy workflow powtarza je na hostowanych Windows,
Ubuntu i macOS; na Windows dołącza pywin32 jako adapter natywnego filesystem,
bez instalacji produktu ani wykonywania lifecycle.

## Późniejsza kwalifikacja na autoryzowanym hoście

Poniższe kroki są planem do osobnej autoryzacji. Ten PR nie zezwala na ich
wykonanie i nie zawiera polecenia uruchamiającego fizyczną próbę.

1. Zaimplementować oraz zrecenzować brakujące obserwatory bootu, pochodzenia
   startu, ciągłego życia procesu i zakresu monotonicznego. Zdefiniować ich
   źródła zaufania, politykę suspend/resume, failure modes i format wiązania
   próby. Uzupełnić syntetyczne testy każdej obserwacji, replay, błędów wejść,
   przerwań i publikacji. Same testy syntetyczne nie są dowodem fizycznym.
2. Ukończyć wymagane warunki Stage 9 i legal production enrollment przez
   istniejące autorytety. Zachować blokady `active_hba`, runnera i pozostałych
   brakujących mechanizmów do czasu dostarczenia rzeczywistych dowodów.
   Nie zmieniać wymagań, aby dopasować je do dostępnego markera.
3. Zgodnie z
   [read-only preflight](windows_stage10_readonly_preflight.md) zatwierdzić
   izolację i tożsamość dedykowanego hosta/runnera oraz zaufaną trasę dispatch.
   Środowisko PR ani etykiety runnera nie stanowią autoryzacji hosta.
4. Ustalić przez uwierzytelnione metadane GitHub dokładną rewizję, CI run/attempt,
   artifact ID i niezależne digesty. Użyć
   [`windows_stage10_artifact_preflight`](../deployment/windows_stage10_artifact_preflight.py)
   do weryfikacji MSI, manifestu, receipt i istniejącego Production Trust.
   Osobno rozwiązać instalację zatwierdzonego produktu na docelowej maszynie;
   hosted clean-install ani historyczny receipt nie dowodzą tej instalacji.
5. W osobnym, zatwierdzonym zakresie potwierdzić rzeczywisty enrolled runtime
   przez istniejący kwalifikator live. Każda wymagana odmowa lub brak władzy
   zatrzymuje dalsze kroki. Nie uruchamiać mutujących poleceń instalatora pod
   nazwą read-only qualification.
6. Dopiero po przeglądzie implementacji i oddzielnej autoryzacji boot/soak
   wykonać próbę na zatwierdzonym hoście. Powiązać obserwacje przed/po z jedną
   tożsamością próby, zatwierdzonym produktem, procesem i kontekstem CI.
   Zbierać ciągły dowód życia procesu oraz świeże dowody gotowości.
7. Zweryfikować każdą z ośmiu deklaracji osobno, sprawdzić integralność całego
   zakresu i publikację końcową. Niezależny odbiorca ponownie sprawdza źródła
   i wiązania. Jakikolwiek brak utrzymuje odmowę; `PASS` nie wynika z samego
   zakończenia skryptu, upływu czasu ani zgody operatora.

## Znaczenie walidacji tego PR

Testy tego PR weryfikują odmowę nieuzasadnionego `PASS`, jawny stan wszystkich
deklaracji, odrzucanie syntetycznych/historycznych receipt oraz brak uruchomienia
ścieżki reboot/soak. Test pozytywny dla poprawnej odmowy nie jest pozytywnym
testem fizycznej kwalifikacji. Nie istnieje zaakceptowana ścieżka pozytywnego
receipt v1; przyszłe jej dodanie wymaga implementacji obserwatorów, pełnych
testów i osobnego przeglądu granicy zaufania.

`WINDOWS_LIVE_LIFECYCLE_QUALIFIED=NO`

`STAGE10_PRODUCTION_READY=NO`
