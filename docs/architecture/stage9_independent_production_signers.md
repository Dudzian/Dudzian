# Stage 9 — S9-RPI-02: independent production signer services

## Zakres i decyzja

Zmiana kontynuuje S9-RPI-01 z `main` po PR #3151 (`41e11ebc`). Przygotowuje
dwa odrębne procesy Linux: `RootProofSigner` i `HistoryAttestationSigner`.
Ich jedyne operacje IPC to `DIAGNOSTIC` i `QUALIFY`. Nie istnieje operacja
`SIGN`, provisioning kluczy, enrollment, aktywacja ani produkcyjny BIND.
Publiczna emisja Root Proof pozostaje **NO-GO**; production `commit()` nadal
odmawia, a zamrożone kontrakty i kanoniczne statusy readiness nie są zmieniane.

Powstają trzy różne rodzaje wyniku: implementacja transportu, obserwacja
tożsamości procesu i diagnostyka custody. Żaden samodzielnie nie jest
kwalifikacją fizycznej instalacji ani upoważnieniem do emisji.

Podstawą były rzeczywiste źródła `production_root_lifecycle.py`,
`postgresql_production_root_lifecycle_schema.py`,
`postgresql_production_root_lifecycle_qualification.py`,
`local_signing_custody.py`, `security/keyring_storage.py` oraz dokumentacja
[S9-RPI-01](stage9_production_root_proof_lifecycle_isolation.md).
Codebase Memory MCP nie był callable; celowane wyszukiwania i odczyty źródeł
nie są deklaracją kompletności nieobecnego indeksu.

## Instalacyjna separacja usług

| Właściwość | RootProofSigner | HistoryAttestationSigner |
| --- | --- | --- |
| Rola instalacyjna | `root_signer` | `history_signer` |
| Linux user i primary group | `cryptohunter-root-proof-signer` | `cryptohunter-history-attestation-signer` |
| Prywatna konfiguracja | `/etc/cryptohunter/root-proof-signer/custody.json` | `/etc/cryptohunter/history-attestation-signer/custody.json` |
| Custody directory | `/var/lib/cryptohunter/root-proof-signer` | `/var/lib/cryptohunter/history-attestation-signer` |
| Dedykowany socket directory | `/run/cryptohunter/root-proof-signer` | `/run/cryptohunter/history-attestation-signer` |

UID/GID są rzeczywistymi, odrębnymi identyfikatorami przypiętymi w chronionym
`/etc/cryptohunter/root-proof-issuer/installation.json`; nazwy z tabeli są
konwencją dostarczonych assetów. Instalacja przypina także pozostałe pięć
principals S9-RPI-01. Procesy nie mają wspólnej supplementary service group.
Konfiguracje custody mają root ownership i mode `0640`; ich bezpośrednie
katalogi są `root:role_gid 0750`. Inny signer i requester nie otrzymują
dostępu. Publiczny manifest topology jest root-owned i chroniony przed
modyfikacją; nie zawiera sekretów. Nie ma caller override ścieżki,
profilu, DSN, principal ani socketu.

Custody directories są `role_uid:role_gid 0700`, metadata `0600`, a wszystkie
wyższe katalogi root-owned bez group/world write. Extended POSIX ACL nie są
akceptowane. Runtime montuje własny magazyn read-only i ukrywa magazyn oraz
konfigurację drugiej roli. Nie tworzy lock file, keyring index, metadata,
klucza ani katalogu podczas odczytu brakującego credential.

W chronionym `installation.json` pola `signer_sockets` dla tych units muszą
wskazywać `/run/cryptohunter/root-proof-signer/signer.sock` oraz
`/run/cryptohunter/history-attestation-signer/signer.sock`. Nie aktualizujemy
manifestu automatycznie. Nowy port SQL `observe_signer_custody(text,text)`
zmienia oczekiwany reviewed catalog; starsza instalacja odmawia kwalifikacji
do czasu osobno zatwierdzonej aktualizacji i przypięcia katalogu. Nie ma
runtime re-pin ani migracji wykonywanej przez requestera lub signera.

Dedykowany socket directory jest `signer_uid:issuer_gid 02710`. Setgid
przenosi issuer GID na socket `signer_uid:issuer_gid 0660`, bez dopisywania
signera do grupy issuera i bez `CAP_CHOWN`. Issuer ma traversal, ale nie może
tworzyć, usuwać ani zastępować endpointu. Wyższa ancestry jest root-owned
bez group/world write. Nie używamy `.socket` activation: poświadczenia
listeneru utworzonego przez root nie byłyby poświadczeniami procesu signera.

Units w `deployment/systemd/` mają osobne mount, IPC i network namespaces,
`NoNewPrivileges`, puste capabilities, `ProtectSystem=strict`, `ProtectHome`,
AF_UNIX only i zakaz tworzenia dalszych namespaces. Nie mają sekcji
`[Install]`, zależności uruchamiającej produkcję ani automatycznych restartów.
Pliki `sysusers.d/`, `tmpfiles.d/` i przykłady JSON są wejściami do przeglądu
administratora. Ta zmiana ich nie instaluje ani nie uruchamia usług.
Przykładowy trust domain wymaga zastąpienia wartością z reviewed installation.
Nie należy kopiować przykładu jako gotowej konfiguracji produkcyjnej.

## IPC i tożsamość procesu

`bot_core/production_signer_ipc.py` implementuje wersjonowany protokół
ograniczony do JSON diagnostycznego. Zduplikowane i dodatkowe pola,
nieprawidłowe typy, obca rola i operacje poza allowlist odmawiają. Ramka ma
limit 16 KiB i wspólny deadline I/O 3 sekundy; klient nie ponawia operacji.
Dokładna rola, manifest instalacji i odpowiedź należą do tego samego exchange.

Socket i każdy element ścieżki są otwierane deskryptorami z `O_NOFOLLOW`.
Kontrole obejmują typ obiektu, UID/GID, mode, brak ACL, link count oraz
ponowne sprawdzenie inode i metadata. Połączenie przechodzi przez przypięty
parent descriptor, a nie ponownie rozwiązaną dowolną ścieżkę.
`SO_PEERCRED` po obu stronach wymaga exact peer PID/UID/GID, innego procesu
i właściwej roli. `pidfd` oraz przypięte `/proc/<pid>` wiążą obserwację
z żyjącym procesem, jego start time, real/effective/saved IDs i groups.
Zmiana socketu, śmierć procesu lub zerwane połączenie unieważniają wynik.
Nie wolno używać poprzedniej odpowiedzi po restarcie jako aktualnej
kwalifikacji nowego procesu.

Odczyt `/proc/<pid>/ns` między różnymi UID może odmówić przez Linux ptrace
access checks. Usługi nie dostają `CAP_SYS_PTRACE`. Brak tej obserwacji
pozostawia jawną blokadę kwalifikacji namespaces; nie jest domyślnym PASS.
Same dyrektywy systemd nie dowodzą, że fizyczne procesy dostały odrębne
namespaces i że nie istnieje alternatywna ścieżka dostępu do kluczy.

Stale socket nie jest automatycznie nadpisywany. Zwykłe zamknięcie usuwa
wyłącznie własny, nadal dokładnie przypięty obiekt. Po crashu administrator
musi osobno rozstrzygnąć pozostały endpoint; startup odmawia przy istniejącym
obiekcie. CLI obsługuje SIGTERM/SIGINT przez stop event i cleanup serwera.
`Restart=no` zapobiega pętli automatycznych prób. Deadline klienta nie jest
dowodem ograniczonego czasu wykonania blokującego native-keyring API;
responsywność tego backendu wymaga odrębnej kwalifikacji. Zabicie procesu
podczas takiego odczytu pozostawia stale endpoint i blokuje ponowny startup.

## Custody i niezależny PostgreSQL lifecycle

`bot_core/production_signer_custody.py` korzysta z istniejącego formatu
`local_signing_custody` oraz read-only dostępu do istniejącego native keyring.
Sprawdza canonical credential identity, provider/semantic role, trust domain,
namespace, public key fingerprint, key version, generation i lifecycle.
Obce lub zamienione credential, materiał i niespójny state/generation
odmawiają. Odczyt istniejącego seed i wyprowadzenie public key potwierdzają
lokalną zgodność z publicznym metadata, bez wystawiania podpisu i sekretu.
Metadata i konfiguracja są ponownie sprawdzane po odczycie.

Ta obserwacja nadal pochodzi z procesu signera. Nie dowodzi niezależnie
wyłącznej kontroli prywatnego klucza ani odmowy dostępu drugiej roli do
native keyring. Wynik zawsze zachowuje `private_control_status=BLOCKED`
oraz blokady niezależnego private-control attestation i nieudowodnionej
izolacji native keyring. Niedostępny, zablokowany lub nieodpowiedni native
backend odmawia; nie przełącza się na magazyn plikowy, mock ani nowy klucz.

Issuer w `ProductionRootLifecycleBoundary` porównuje publiczne fakty
z niezależnym, zainstalowanym PostgreSQL lifecycle authority, używając
istniejącej kwalifikacji physical identity, catalog i direct peer login.
Deklarowana przez signera rola, identity, materiał, generation i lifecycle
nie są samodzielnym dowodem prawdy. Brak PostgreSQL, zmiana manifestu,
niezgodne facts albo brak właściwego fence nie daje pozytywnego wyniku.
`PENDING_CLOSED` nie staje się `ACTIVE` przez diagnostykę. `VERIFY_ONLY`
i `REVOKED` nie upoważniają do nowego podpisu. Ta zmiana nie otwiera fence,
nie mutuje lifecycle i nie nadaje signerom dostępu do authority SQL.

Entrypoint ma jedynie dwie formy z fixed instalacją:

```text
python -m deployment.production_signer_service root_signer --check
python -m deployment.production_signer_service history_signer --serve
```

`--check` wypisuje ograniczony publiczny JSON i zwraca kod `2` dla `BLOCKED`.
`--serve` uruchamia wyłącznie port diagnostyczny wskazanej roli. Błędy native
backend nie są przepisywane do publicznego outputu jako niekontrolowany tekst.
Nie ma override konfiguracji, sign, generowania klucza ani auto-retry.

## Walidacja i pozostałe blokady

Testy `test_production_signer_ipc.py` używają rzeczywistych procesów Linux
tam, gdzie środowisko pozwala wykonać system calls. Testy custody i deployment
obejmują odmowę malformed input, obcej roli, niewłaściwych uprawnień,
zamiany materiału i generation, niedostępności backend oraz brak SIGN
i caller overrides. Testy izolowanego PostgreSQL 16 i celowane regresje
S9-RPI-01 sprawdzają fakty implementacji oraz SQL, nie autoryzację produkcji.
Testy z kontrolowanymi stubami są dowodem logiki walidacji; nie są dowodem
physical custody ani separacji kont na zainstalowanym hoście.

Dokładne wykonane polecenia i wyniki testów należą do opisu PR. Nie wykonano
instalacji usług, native-keyring qualification na fizycznym hoście,
produkcyjnego provisioning, kluczy, enrollmentu, BIND ani Stage 10. Nie
deklarujemy PASS dla niewykonanej kwalifikacji. Pełne wielogodzinne CI nie
jest lokalną bramką oczekiwania; jeżeli nadal trwa, PR zachowuje
`FULL_CI_PENDING`, a istniejące workflow i progi pozostają aktywne.

Do publicznej emisji pozostają co najmniej niezależne private-key control
i cross-role native custody attestation, kwalifikacja fizycznego deploymentu
(w tym namespaces i wszystkie ścieżki logowania PostgreSQL), protokół
aktywacji/sukcesji i trust activation oraz produkcyjny semantic issuer.
Nierozstrzygnięty konflikt transaction separation z atomic BIND z S9-RPI-01
pozostaje blokadą kontraktową. Diagnostic IPC i dodatkowy odczyt SQL tego
konfliktu nie rozwiązują. Rekomendacja emisji pozostaje **NO-GO**.

### Host qualification limits

The IPC identity pin requires Linux 6.5 or newer and working `SO_PEERPIDFD`
(option 77); a header constant alone is insufficient. A missing or denied
socket-associated pidfd fails closed without a `pidfd_open(pid)` fallback.

Native SecretService access, its bus/session and the repository HWID envelope
must be qualified inside each installed private network namespace. HWID
collection includes namespace-visible interfaces and `uuid.getnode()`; an
existing host-provisioned envelope may be unavailable or unstable across
service restarts. These assets do not establish that physical qualification.
The issuer rejects observed sharing with its own or the other signer's
mount, IPC or network namespace. Unreadable cross-UID namespace evidence stays
`BLOCKED`, including the cross-signer comparison.
