# Stage 9 Linux — rzeczywiste adaptery usług i granice kwalifikacji

Punktem odniesienia jest `main` po PR #3233
(`e0a005e616881692f8401d57dbc5de3486b3b9ef`). Inwentaryzacja obejmuje
bezpośredni odczyt modułów kolektora, walidatora i tożsamości wydania,
czterech jednostek `deployment/systemd`, ich entrypointów, klienta freshness
oraz granic Source/CCA. Codebase Memory MCP i CLI były niedostępne; nie
wykonano kontroli pokrycia indeksu. Ustalenia dotyczą sprawdzonego zakresu
źródeł, a nie kompletnego spisu procesów istniejącej instalacji.

## Mapa rzeczywistych komponentów

| Komponent | Wymagany kontrakt | Istniejąca implementacja | Luka i wymagany dowód |
| --- | --- | --- | --- |
| `cryptohunter-freshness-verifier.service` | `cryptohunter.verifier.readonly-health.v1`, `cryptohunter.ipc.server-authentication.v1`, `cryptohunter.postgresql.verifier-peer-and-denials.v1` | `freshness_verifier_service.main()` buduje rzeczywisty `FreshnessSemanticVerifier`, resolver i preparer z `PostgreSQLVerifierConnection`. Główny protokół przyjmuje trzy uwierzytelnione obiekty i tworzy przygotowanie w PostgreSQL. | Potrzebna odczytowa wymiana w tym samym procesie, przypięcie obu stron i rzeczywista odmowa. Własna obserwacja PostgreSQL musi zostać skorelowana z niezależnie obserwowanym procesem, backendem i skutecznym HBA/ident. |
| Runtime AccountGenesis jako biblioteka produktu | `cryptohunter.runtime.readonly-health.v1`, `cryptohunter.ipc.client-authentication.v1`, `cryptohunter.postgresql.runtime-peer-and-denials.v1` | `ProductionLocalFreshnessAuthority` wykonuje prawdziwy AF_UNIX do verifiera, następnie SERIALIZABLE `freshness_authority.compare_and_advance` jako stała rola `freshness_runtime`. To rzeczywisty kod produktu. | W sprawdzonym zestawie jednostek nie ma entrypointu samodzielnej usługi runtime. Potrzebny jej rzeczywisty właściciel procesu oraz zatwierdzona jednostka, executable i cgroup. Proces uruchomiony tylko do sondy nie jest taką usługą. |
| RootProofSigner i HistoryAttestationSigner | Własne kontrakty zainstalowanych signerów | Dwie jednostki uruchamiają `deployment.production_signer_service` z odrębnymi rolami i kontami. Dozwolone są tylko `DIAGNOSTIC` i `QUALIFY`; odpowiedź zachowuje `BLOCKED`. Istniejące `_PinnedPeer` używa SO_PEERCRED i SO_PEERPIDFD. | To inne role niż runtime/verifier. Test signera nie może spełnić kontraktu freshness. Diagnostyka nie uzyskuje SIGN ani dostępu do produkcyjnych danych do podpisu. |
| RootProofIssuer check | Kwalifikacja lifecycle RootProofIssuer | Jednostka oneshot uruchamia `deployment.production_root_proof_issuer --check`. `ProductionRootLifecycleBoundary.commit()` zachowuje zamrożoną odmowę produkcyjną. | Check nie jest daemonem runtime ani endpointem Source/CCA. Adaptery platformowe nie zmieniają jego uprawnień. |
| Proponowane Source/CCA: `root_source`, `history_source`, `cca` | `CryptoHunter.SourceCCAInstallationV1` i izolacja trzech odrębnych autorytetów | `stage9_trusted_installation_runtime.observe_service()` dwukrotnie koreluje systemd, NSS, procfs, executable i cgroup. Preflight jawnie opisuje topologię jako proponowaną. Identity gate zachowuje `NOT_QUALIFIED`. | Obserwatory i procesy `TEST ONLY` nie implementują produkcyjnych autorytetów. Nadal wymagane są niezależne zatwierdzenie, socket-bound peer, skuteczna izolacja, provenance kodu, custody i trwałość Source lease/CCA. |

Profil wydania w `linux_release_identity.py` wymaga **obu** ról
runtime/verifier. Podpisany certyfikat wiąże dokładną rewizję, programy,
kontrakty i trzy pakiety. Nie dowodzi wykonania kontraktów. Nazwy podobne
do wymaganej usługi ani poprawnie podpisany raport nie zastępują produktu.

## Odczytowa gałąź rzeczywistego verifiera

Adapter `bot_core/linux_service_ipc.py` dodaje zamkniętą operację
`READONLY_QUALIFY` do istniejącego socketu verifiera. Jest ona dostępna dla
kontrolowanego kolektora root i nie korzysta z operacji przygotowania
kandydata. Główny protokół nadal wymaga dozwolonego UID runtime. Nie powstaje
interfejs podpisywania, administracji, surowego SQL ani wyboru roli bazy.

Uruchomienie adaptera wymaga jawnego, chronionego pliku
`/etc/cryptohunter/linux-qualification.json` z wersją i identyfikatorem
instalacji. Konfiguracja nie pochodzi z żądania klienta. Request wiąże
operację, wersję, installation_id, challenge_id i świeży nonce. Obserwacje
obejmują prawdziwą odpowiedź aplikacji, jądrową tożsamość partnera oraz
kontrole procesu i jego czasu życia. Brak wymaganej obserwacji powoduje
odmowę albo zachowanie jawnego ograniczenia; nie jest uzupełniany deklaracją.

Wynik `PARTIAL_VERIFIER_OBSERVATION` potwierdza zakres wykonany na verifierze.
Nie dowodzi autoryzowanego połączenia **rzeczywistego runtime**, jego health
ani pełnej macierzy odmów. Kontrola powtórzenia challenge/nonce w pamięci
procesu nie jest trwałą ochroną przed replay po restarcie; ta luka pozostaje
oddzielną blokadą. Odmowa aplikacji wobec obcej tożsamości musi być odróżniana
od błędu ramki, timeoutu i odmowy klienta.

## Połączenie PostgreSQL należące do usługi

`bot_core/linux_service_postgresql.py` obserwuje przekazane, rzeczywiste,
bezczynne połączenie należące do bieżącego procesu. Nie otwiera osobnej
sesji jako administrator, nie zmienia UID/GID i nie udostępnia SQL klientowi.
Właściwa aplikacja wywołuje adapter na swoim połączeniu; kolektor następnie
porównuje obserwacje z niezależnie zatwierdzonym procesem i instalacją.

Stała macierz wykonuje odczytową transakcję, odczytuje `system_user`, role,
backend PID, bazę i metadane lokalnego transportu. `SELECT 1` potwierdza
rzeczywistą dozwoloną wymianę z bazą, ale **nie potwierdza semantycznej
operacji produktu**. Próby `SELECT ... LIMIT 0` sprawdzają ACL bez pobierania
wierszy katalogu haseł lub prywatnych tabel. Odmowa musi mieć rzeczywisty
SQLSTATE `42501`; inny błąd, brak tabeli albo utrata połączenia nie zastępują
odmowy uprawnień. Odczytowa transakcja i jej wycofanie chronią trwały stan.

Mapy `pg_ident` i skuteczne HBA wymagają niezależnej korelacji; sam raport
procesu nie dowodzi zatwierdzenia plików konfiguracji. Oddzielna administracyjna
obserwacja kolektora pozostaje oddzielona od dowodu uprawnień usługi.
Pełna macierz operacji produktu i połączenie rzeczywistego runtime pozostają
wymagane do usunięcia odpowiadających im blokad.

## Włączenie do producenta i niezależnej walidacji

Opcjonalny profil `installation.functional_qualification` zawiera dokładnie
`installation_id`. Ten sam identyfikator musi wystąpić w chronionej polityce
usługi; jej dokładne bajty są przypięte w `installation.configuration`.
Nie jest to hash pliku zawierającego własny hash. `NativeDriver` wybiera
jedyną zatwierdzoną jednostkę verifiera i stały socket produktu. Przed i po
wymianie porównuje systemd, PID, UID/GID, czas startu, program i cgroup.
Zamiana jednostki albo procesu wymaga odmowy.

Surowe `functional_observations` są walidowane oddzielnie od pełnej
`ipc_qualification` i `postgres.service_qualification`. Profil funkcjonalny
zachowuje również surowe reguły HBA i mapowania ident obok ich digestów.
Pozwala to ponownie sprawdzić mapowanie bez utożsamienia połączenia
administratora z połączeniem verifiera.

`collect_functional_diagnostic()` i CLI kolektora `--observe-verifier`
zbierają odczyt w odrębnej domenie `STAGE9_PARTIAL_VERIFIER_DIAGNOSTIC`.
Weryfikator `verify_functional_diagnostic()` oraz CLI `verify-diagnostic`
sprawdzają challenge, podpis, host, materiały i surowe relacje. Zwracają
wyłącznie `PARTIAL_BLOCKED`. Nie można użyć takiego rekordu jako siedmiu faz
`verify_bundle()` ani jako wyniku LIVE dla `platform_readiness`.

`collect_phase()` nadal odmawia natywnych faz mutujących, gdy pełny kontrakt
IPC nie został spełniony. `verify_bundle()` nadal wymaga obu pełnych
kontraktów i zatwierdzonego wydania. Diagnostyka VM nie powoduje dodania VM
do polityki fizycznych hostów ani rozluźnienia wymagań istniejącej bramki.

## Zachowane granice i ryzyka regresji

- Nie zastępujemy Source/CCA rolami runtime/verifier ani signerami. Kontrakt
  Source/CCA wymaga odrębnych UID/GID bez grup dodatkowych; grupa IPC freshness
  nie ustanawia wyjątku od tego kontraktu.
- Publiczny interfejs `ProductionLocalFreshnessAuthority` pozostaje jedną
  operacją `authenticate_and_advance`. Test architektury wyklucza dodanie
  ogólnego API health, prepare, SQL lub administracji do tego autorytetu.
- `freshness_deployment_qualification.REVIEWED_CODE_DIGESTS` przypina dokładne
  bajty modułów freshness oraz zależności nowych adapterów. Ich zmiany wymagają jawnego przeglądu,
  aktualizacji pinów i regresji istniejącej kwalifikacji.
- Hash `/proc/PID/exe` procesu Python identyfikuje interpreter. Nie ustanawia
  samodzielnie tożsamości modułów, importów i bibliotek; wymagane jest związanie
  z zatwierdzonym wydaniem i chronionymi plikami kodu.
- Brak odczytu executable lub cgroup, utrata procesu, podmiana PID, ponowne
  użycie PID i TOCTOU nie mogą zostać zamienione na pozytywny dowód.
- Weryfikator paczki ponownie ocenia surowe fakty. JSON z `PASS` ani podpis
  producenta nie usuwają blokady brakującego kontraktu produktu.

## Zakres regresji i wykonania fizycznego

Szybka regresja obejmuje nowe adaptery, moduły `linux_physical_*`, release
identity, freshness semantic verifier, deployment oraz kontrakty architektury
M05. Zachowanie Source/CCA sprawdzają istniejące testy kontraktu instalacji,
runtime, identity gate i odmów. Testy signerów pozostają wymagane tam, gdzie
adapter używa ich istniejącego mechanizmu przypięcia partnera.

Rzeczywiste testy peer PostgreSQL z `isolated_peer_cluster` tworzą konta
systemowe i własny tymczasowy klaster. Ich wykonanie jest dopuszczalne
wyłącznie w odtwarzalnym, izolowanym środowisku opisanym w
[procedurze VM](../runbooks/STAGE9_LINUX_FUNCTIONAL_ADAPTERS_VM.md).
Wynik integracji adaptera nie jest dowodem wydanej instalacji ani fizycznej
kwalifikacji. Zmiany nie ustawiają Stage 9 na COMPLETE i nie zmieniają
`LINUX_PRODUCTION_READY=NO`.

## Regresja CI po PR #3233

Test `test_production_issuance_effects_and_consumers_preserve_local_dispatch_ownership`
odtworzył `issuer transport bypass in deployment.linux_physical_ipc`. Najkrótsza
zaobserwowana ścieżka importów prowadziła przez:

```text
cha_root_proof_attempt_reservation
→ windows_production_cha_account_reservation
→ windows_production_cha_operation
→ windows_production_lppi_authority
→ platforms.windows
→ windows_stage8_postgresql_probe
→ platform_evidence
→ linux_physical_evidence
→ linux_physical_ipc
```

Sonda Windows potrzebowała wyłącznie dwóch identyfikatorów dowodów, lecz
importowała cały moduł producenta platformowego. Analiza forward closure
prawidłowo wykryła poszerzenie zależności lokalnej emisji o transport Linux;
nie stwierdzono wywołania tego transportu z operacji emisji. Przyczyną było
zbyt szerokie sprzężenie modułów przy imporcie stałych.

Stałe przeniesiono do `windows_stage8_evidence_contract.py`, który nie ma
zależności wykonawczych. Sonda i producent korzystają teraz z tego samego
kontraktu danych. Zachowano kompatybilny eksport identyfikatorów producenta.
Nie zmieniono asercji bezpieczeństwa, wyjątków transportowych, ochrony
podpisywania, progów ani analizy AST. Nowe testy sprawdzają czysty kontrakt,
zamknięty zakres zależności sondy i wspólne identyfikatory.

Zidentyfikowano również wcześniejszy drift pinu `account_genesis_freshness_authority.py`: pin w main to `cd508809…`, a bajty pliku w tym samym main mają SHA256 `81cd0df375fe1a48ba943c7645376b4d97350288c0ebe6523207ff90ed22572b`. Tego niezwiązanego pinu ani modułu nie zmieniono w tym PR; pełna istniejąca kwalifikacja deploymentu nadal odmówi i wymaga osobnego przeglądu tych bajtów.
