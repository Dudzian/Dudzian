# Stage 41B — fizyczna separacja syntetycznych kluczy signerów

Stage 41B kwalifikuje testową granicę między procesami `RootProofSigner`
i `HistoryAttestationSigner` na fizycznym Ubuntu 24.04 z systemd jako PID 1.
Nie zmienia statusu Stage 9 (`IN_PROGRESS` / **NO-GO**) ani produkcyjnego
`private_control_status=BLOCKED`. Wynik `STAGE41B_TEST_ONLY_PASS` oznacza
wyłącznie pomyślną kontrolę wszystkich wymaganych granic w jednym wykonaniu
z nowymi, syntetycznymi kluczami. Nie upoważnia do produkcyjnego SIGN, BIND,
provisioningu, enrollmentu ani Stage 10.

## Zakres i model zagrożeń

Badany przeciwnik to proces drugiej roli, działający bez capabilities i bez
możliwości podniesienia uprawnień. Kontrole obejmują rzeczywiste UID/GID,
mount/IPC/network namespaces, DAC, natywny Secret Service, lokalny D-Bus,
wiązanie credential z rolą i tożsamością oraz odrzucanie żądań testowego IPC.
Harness administracyjny i systemd są zaufane; root hosta, kernel, firmware,
złośliwy administrator i pełne przejęcie hosta pozostają poza tym modelem.
Osobne namespaces nie tworzą odrębnych kernelów ani sprzętowego HSM.

Każda rola otrzymuje niezależny losowy Ed25519 seed, losowy master custody,
losowe hasło własnego GNOME keyringu, osobny bus i nowy testowy trust domain.
Materiał nie pochodzi z repozytorium i nie jest pobierany z produkcyjnych
instalacji. Natywny backend to dokładna implementacja
`keyring.backends.SecretService.Keyring` z GNOME Keyring, bez mocka,
fallbacku plikowego lub chained backendu podczas części fizycznej.

Odczyt, format metadata, identity binding i custody korzystają z oryginalnego
`bot_core/local_signing_custody.py` oraz `bot_core/security/keyring_storage.py`.
Test inicjalizuje nowy syntetyczny master wyłącznie w nowym prywatnym busie;
nie naprawia istniejącego magazynu ani nie odblokowuje automatycznej inicjalizacji
mastera w runtime. Brak dostępnego backendu lub protected HWID to BLOCKED.

Produkcyjny `deployment/production_signer_service.py` i
`bot_core/production_signer_ipc.py` nadal obsługują wyłącznie `DIAGNOSTIC`
i `QUALIFY`. Osobny worker Stage 41B podpisuje wyłącznie jednoznacznie
oznaczone dane testowe z bieżącego wykonania. Jego protokół nie jest portem
produkcyjnego signera ani źródłem produkcyjnego attestation.

## Kontrolowany tryb fizyczny

Entrypoint to `scripts/qualify_stage41b_signer_isolation.py`; worker to
`deployment/stage41b_role_worker.py`. Zwykłe pytest sprawdza logikę
klasyfikacji, walidację i ścieżki błędów, bez startowania uprzywilejowanych
usług. Nie należy uruchamiać root-only pytest bezpośrednio na głównym Ubuntu.

```bash
cd /opt/cryptohunter-stage41b
sudo /absolute/path/to/python -B -m scripts.qualify_stage41b_signer_isolation --check
sudo /absolute/path/to/python -B -m scripts.qualify_stage41b_signer_isolation --execute
```

Użyj uprzednio przygotowanego, kontrolowanego fizycznego hosta kwalifikacyjnego
z Ubuntu 24.04, działającym systemd i lokalnym GNOME Secret Service. Preflight
odrzuca kontener i maszynę wirtualną przez `systemd-detect-virt`; nie interpretuje
przebiegu w VM jako fizycznej kwalifikacji. Python musi mieć
zainstalowane zależności CryptoHuntera oraz `keyring` i `secretstorage`.
Checkout oraz interpreter i jego zależności muszą pochodzić z wcześniej
przejrzanej, root-owned instalacji, bez zapisu przez role testowe. Harness
odmawia wykonania z domowego lub tymczasowego checkoutu. Także `--check`
wymaga kontrolowanego uruchomienia jako root, aby ocenić protected HWID
i stan jednostek; nie jest to uruchomienie pytest. Flaga `-B` wyłącza
zapis cache bytecode w preflight. Wykonanie nie instaluje zależności,
nie uruchamia produkcji i nie wykonuje
`systemctl enable`. Nie kieruj go do hosta z produkcyjnymi sekretami.

`--check` wykonuje preflight bez tworzenia kont, plików credential, units,
keyringów lub kluczy. Nawet pozytywny preflight nie daje
`STAGE41B_TEST_ONLY_PASS`. `--execute` ponawia sprawdzenia i uruchamia osobne,
tymczasowe units systemd z odrębnymi DynamicUser UID/GID, prywatnymi
namespaces i `NoNewPrivileges`. Katalogi testowe, DynamicUser i opis jednostki
zawierają losowy identyfikator wykonania. Dwie jednostki transient używają
dokładnych nazw `cryptohunter-root-proof-signer.service` oraz
`cryptohunter-history-attestation-signer.service`, ponieważ oryginalny reader
custody akceptuje wyłącznie ich stałe ścieżki credential.

Przed wykonaniem nazwy te muszą być niezainstalowane, niezamaskowane,
niezaładowane i nieaktywne. Guard obejmuje także issuer-check oraz istniejące
kanoniczne katalogi credential. Każda kolizja powoduje odmowę; harness nie
zatrzymuje produkcji, nie nadpisuje ani nie przejmuje obcych units. Ten test
nie może działać obok zainstalowanych produkcyjnych signerów. `systemd-run`
tworzy wyłącznie tymczasowe units pod `/run/systemd/transient`, bez instalacji
assetów `deployment/systemd` i bez `enable`.

Oryginalny reader custody oczekuje dwóch stałych nazw jednostek w ścieżce
`/run/credentials/<production-unit>/custody-product-uuid`. Test dostarcza
oryginalny kernel DMI UUID przez standardowe `LoadCredential`; systemd zarządza
tymczasowym read-only credential mount dla kontrolowanej jednostki testowej.
Nie zmienia providerów HWID ani nie tworzy aliasu alternatywnej ścieżki.
Brak pozostałości kanonicznych katalogów credential musi zostać potwierdzony
także po sprzątaniu. Prywatne konfiguracje i dane GNOME pozostają w losowych
katalogach bieżącego wykonania; produkcyjne pliki instalacyjne nie są otwierane.

## Interpretacja wyników

Każda wymagana granica musi mieć własny wynik `PASS`, `FAIL` albo `BLOCKED`.
`FAIL` oznacza wykazane naruszenie lub niezgodność wymaganej właściwości;
`BLOCKED` oznacza brak warunków lub dowodu pozwalającego ją ocenić. Timeout,
martwy proces, brak właściciela nazwy Secret Service lub niekompletna
odpowiedź nigdy nie stają się pozytywnym wynikiem.

| Obserwacja | Znaczenie | Warunek pozytywnej oceny |
| --- | --- | --- |
| `EACCES` / `EPERM` przy obcym istniejącym pliku lub socket | Odmowa uprawnień | Żywy właściciel i pozytywny dostęp własnej roli |
| `ENOENT` w prywatnym mount namespace | Ścieżka niewidoczna w tej przestrzeni | Istnienie obiektu z perspektywy hosta oraz pozytywne own-side kontrole obu usług przed i po próbie |
| Brak procesu, bus ownera, odpowiedzi lub timeout | Usługa niedostępna | `BLOCKED`, bez wniosku o izolacji |
| Odczyt obcego pliku, autoryzowane połączenie z obcym busem lub odzyskany sekret | Naruszenie granicy | `FAIL` |
| Odrzucenie obcej identity/roli/reference przez oryginalne custody | Odmowa na istniejącym porcie | Osobno potwierdzona poprawność własnej identity i własnego odczytu |

Sama obecność pliku socket i samo `systemctl is-active` nie dowodzą
działającego GNOME Secret Service. Dla obu ról wymagany jest pozytywny
kontakt z własnym D-Bus, owner `org.freedesktop.secrets` i odczyt własnego
credential przez oryginalne custody. Kontrole te obejmują okno testowania
obcych magistral. Ukrycie ścieżki i DAC są oddzielnymi kontrolami; ukryta
ścieżka nie zastępuje sprawdzenia rzeczywistej odmowy uprawnień.

Podpis własnego komunikatu musi weryfikować się własnym kluczem publicznym,
a klucze publiczne obu ról muszą być różne. Identity, semantic role,
provider role, trust domain i fingerprint materiału muszą odpowiadać
oryginalnym regułom custody. Żądania obcej roli, tożsamości, operacji oraz
nieautoryzowanego procesu muszą odmawiać. Odpowiedzi, przechwycone ramki
kontrolnego IPC Stage 41B, wyniki i dostępne logi nie mogą zawierać prywatnego seed ani jego
sprawdzanych reprezentacji. Test nie drukuje materiału przy wykryciu wycieku.

`artifact_secrecy` skanuje transcript testowego control socket oraz odczytane
w chwili kontroli wpisy journald obu units wewnątrz workerów znających własny
losowy materiał. Jest to kontrola dostępnych artefaktów przed zatrzymaniem
jednostek, bez pełnego postmortem audytu późniejszych logów hosta.
Nie przekazuje seed ani mastera harnessowi w celu skanowania. Stdout i stderr
testowego drzewa procesów trafiają do `null`, a core dumps są wyłączone.
Nie przechwytuje całej natywnej magistrali ani każdego pakietu D-Bus:
Secret Service musi przekazać własny sekret uprawnionemu procesowi w obrębie
jego zaufanej roli. Kontrola dotyczy niedopuszczenia materiału do publicznego
control IPC, wyników i dostępnych logów; nie deklaruje, że sekret nie występuje
w autoryzowanym, wewnętrznym protokole natywnego keyringu.

Wynik akceptacji jest obliczany dopiero po zakończeniu wszystkich kontroli
i sprzątania. Brak wymaganej granicy, wynik inny niż PASS, przerwany przebieg
lub niesprzątnięty zweryfikowany zasób wyklucza `STAGE41B_TEST_ONLY_PASS`.

## Pochodzenie zasobów i sprzątanie

Harness tworzy wyłącznie zasoby oznaczone identyfikatorem tego wykonania.
Przed zatrzymaniem jednostki ponownie sprawdza jej tożsamość i pochodzenie;
nazwa nie jest wystarczającym dowodem własności. Sprzątanie odbywa się po
sukcesie, błędzie i obsługiwanym sygnale. Obejmuje procesy w testowym cgroup,
tymczasowe jednostki, sockety, testowe keyringi i katalogi bieżącego wykonania.
Nie usuwa istniejących obiektów ani plików o niepotwierdzonym pochodzeniu.

Przy odmowie sprzątania należy zachować wynik BLOCKED i metadane wskazujące
zasób wymagający przeglądu administratora. Nie wykonuj zbiorczego
`rm -rf /run/cryptohunter*`, zatrzymywania wszystkich units z prefiksem ani
resetowania produkcyjnych keyringów. Nagłe odcięcie zasilania, kernel panic
i `SIGKILL` nie uruchamiają kodu `finally`; katalogi `/run` są nietrwałe,
ale ewentualne pozostałości przed rebootem wymagają osobnego przeglądu
pochodzenia. Ponowne wykonanie nie przejmuje takich zasobów.

## Różnica między dowodem syntetycznym i produkcyjnym

| Dowód Stage 41B | Pozostałe wymaganie produkcyjne |
| --- | --- |
| Nowe testowe klucze w osobnych żywych GNOME keyringach | Niezależne potwierdzenie wyłącznej kontroli konkretnych produkcyjnych kluczy oraz pełnego cyklu ich życia |
| Oryginalne custody i realny host HWID | Kwalifikacja zatwierdzonej instalacji, jej manifestów, konfiguracji, kont i trwałych secretów po reboot |
| Tymczasowe procesy systemd z odrębnymi UID/GID i namespaces | Dowód dotyczący dokładnych zainstalowanych produkcyjnych procesów i wszystkich alternatywnych ścieżek dostępu |
| Odmowa testowanych portów plikowych, D-Bus i custody | Analiza kompletności portów produkcyjnych, recovery, backupu, administracji i credential exportu |
| Podpis komunikatu testowego i publiczna weryfikacja | Produkcyjny protokół autoryzacji podpisu, lifecycle/fencing, aktywacja/sukcesja i trust activation |
| Publiczny raport bez sprawdzanych reprezentacji seed | Produkcyjna polityka retencji logów, crash/core dumps, pamięci i telemetrii całego wdrożenia |
| Pomyślne sprzątanie testowych zasobów | Niezależna operational qualification i niezależny authority zatwierdzający attestation |

Raport powstaje na tym samym zaufanym hoście administracyjnym, który
przygotowuje test. Nie jest niezależnie podpisanym private-key control
attestation. Różne klucze publiczne dowodzą rozdzielności par Ed25519;
nie dowodzą same w sobie, że żadna inna kopia seed nie istnieje.
Sprawdzenie kilku reprezentacji wycieku nie jest dowodem nieistnienia
każdego możliwego kanału bocznego lub dowolnego kodowania materiału.

## Niezmienione blokady Stage 9 GO

Autorytatywny status pozostaje w
[`deployment/stage9_current_status.json`](../../deployment/stage9_current_status.json).
Stage 41B nie zmienia żadnego z jego pól: Stage 9 jest `IN_PROGRESS`,
semantic issuer jest `NOT_IMPLEMENTED / BLOCKED`, production preaccount
credentials są `NOT_PROVISIONED`, Root Proof jest `NOT_ISSUED`, legal
production enrollment jest `NOT_PERFORMED`, a
`production_provisioning_ready=false`. Stage 10 live pozostaje `NOT_STARTED`.

Nadal wymagane są niezależne private-key control attestation i kwalifikacja
produkcyjnej separacji native custody, zatwierdzona instalacja oraz
kwalifikacja wszystkich ścieżek uwierzytelnienia PostgreSQL, produkcyjna
aktywacja i sukcesja, trust activation i semantic issuer. Kierunek A + C
dla atomowego BIND został zatwierdzony 2026-10-18; jego
[decyzja właściciela](decisions/stage9_root_proof_transaction_authority_owner_approval_v1.md)
nadal zapisuje `CONTRACT_ADDENDUM_INTEGRATION_PENDING` /
`PRODUCTION_NO_GO`. Stage 41B nie realizuje versioned contract integration,
produkcyjnej transakcji issuera ani jej kwalifikacji. Historyczne ograniczenia
i aktualizacja decyzji są opisane w
[ADR S9-RPI-04](stage9_root_proof_transaction_authority_reconciliation.md),
[S9-RPI-01](stage9_production_root_proof_lifecycle_isolation.md) i
[S9-RPI-02](stage9_independent_production_signers.md). Jest to lista
powiązanych blokad potwierdzonych w źródłach, nie deklaracja kompletnego
zamknięcia nadrzędnych kontraktów Stage 9.

## Walidacja zmian

Harness raportuje 19 wymaganych granic z wynikiem dla każdej z nich, również
gdy preflight blokuje wykonanie. W środowisku implementacji `--check` zakończył
się `BLOCKED` z powodem `UBUNTU_24_04_REQUIRED`; nie wykonano `--execute`
i nie przyznano `STAGE41B_TEST_ONLY_PASS`.

Testy regresyjne Stage 41B są testami logiki i odmów, nie fizycznym dowodem.
Dokładne polecenia, wyniki, pominięcia i stan fizycznego wykonania należą do
opisu PR. Zachowane są dotychczasowe bramki, baseline skanerów, konfiguracje
bezpieczeństwa i progi jakości. Przed PR wymagane są Ruff, właściwe regresje,
skonfigurowany mypy Linux/Windows oraz dokładne
`python -m pre_commit run ruff-format --all-files`; pliki `deployment/`
wymagają dodatkowego jawnego sprawdzenia formatu, ponieważ ten hook ich
nie obejmuje. Fizycznego `--execute` nie dodaje się do zwykłego PR CI.
