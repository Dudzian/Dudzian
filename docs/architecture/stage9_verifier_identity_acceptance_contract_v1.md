# S9-ID-01 — kontrakt akceptacji K4-v2 TEST_ONLY, wersja 1.0.0

Data: 6 listopada 2026 r. Baza: `Dudzian/Dudzian`, `main` po PR #3254,
`d86eafbfd06696bdc298703bc154b2019807a26a`.
**DO NIEZALEŻNEJ DECYZJI; NIEAKTYWNY. Stage 9: IN_PROGRESS / PRODUCTION_NO_GO.**
`STAGE9_PRODUCTION_READY=NO`. Decyzja etapu 1: **PENDING_OWNER_ACCEPT**;
tożsamość: **NOT_PROVEN**; autorytet: **BLOCKED**; próby natywne: **NOT_RUN**.

Ten dokument jest pakietem decyzyjnym etapu 1
[ADR](stage9_freshness_verifier_identity_adr.md), a nie jego ACCEPT.
Kandydatem aplikacyjnym jest **istniejący verifier Python**. Nie powstaje klient,
launcher, sonda ani profil seccomp. Rust pozostaje alternatywą wymagającą osobnego
uzasadnienia przewagi i ryzyka migracji według ADR. Merge dokumentu lub zielony
test schematu nie upoważnia do implementacji ani uruchomienia eksperymentu.

## Normatywny inwentarz i wersjonowanie

Poniższy zamknięty inwentarz jest danymi dokumentacji. Odczytują go wyłącznie
statyczne testy architektury; produkcja nie może go wczytywać. Nie jest manifestem
instalacji, pinem, podpisem ani źródłem oczekiwanej tożsamości. Nie zawiera
zatwierdzonych wartości dla przyszłego wdrożenia. Każda zmiana przesłanki,
obowiązkowego przypadku lub oracle wymaga nowej rewizji i ponownej decyzji etapu 1;
nieznana wersja, dodatkowe/powtórzone pola lub pominięcie dowodu są niedopuszczalne.
Rekord przyszłej decyzji musi wskazywać dokładny commit dokumentu i materiałów.

```json
{
  "contract_id": "S9-ID-01/K4-v2",
  "schema_version": 1,
  "revision": "1.0.0",
  "scope": "TEST_ONLY",
  "active": false,
  "production_loadable": false,
  "implementation_authorized": false,
  "stage1_decision": "PENDING_OWNER_ACCEPT",
  "identity_status": "NOT_PROVEN",
  "authority_status": "BLOCKED",
  "stage9_status": "IN_PROGRESS / PRODUCTION_NO_GO",
  "candidate": "EXISTING_PYTHON",
  "base_commit": "d86eafbfd06696bdc298703bc154b2019807a26a",
  "predicate": "ALL_REQUIRED_EVIDENCE",
  "evidence": {
    "E01": ["independent_approval", "installation_manifest", "runtime_application_closure", "protected_bytes"],
    "E02": ["system_bus", "name_owner_credentials", "system_manager_identity", "manager_policy"],
    "E03": ["effective_unit", "exec_start_pre", "exec_start", "environment", "drop_ins", "no_delegation"],
    "E04": ["uids_gids_groups", "main_pid", "exec_main_pid", "cgroup", "start_ticks", "boot_id", "invocation_id"],
    "E05": ["connected_af_unix", "so_peercred", "socket_bound_so_peerpidfd", "pidfd_liveness"],
    "E06": ["socket_path", "ancestor_handles", "socket_inode", "owners_modes_acl", "namespace_mount_binding"],
    "E07": ["startup_policy", "protection_from_birth", "no_prior_tracer", "inheritance_inventory", "native_startup_qualification"],
    "E08": ["reviewed_startup_order", "seal_before_listen", "all_threads_abi", "no_exec_fork_fd_delegation", "native_continuity_qualification"],
    "E09": ["pre_payload_recheck", "post_response_recheck", "epoch_revocation", "deadline_no_retry", "independent_evidence_binding"]
  },
  "mandatory_tests": {
    "identity": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N09", "N10", "N11", "N12", "N13", "N14", "N15"],
    "authority": ["N16", "N17", "N18"]
  },
  "owner_decisions": ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"],
  "forbidden_substitutes": ["SERVER_HASH", "READY_FILE", "SECCOMP_MODE_ONLY", "PIDFD_OPEN_FALLBACK", "EXPECTED_AS_OBSERVED"]
}
```

## Stan źródeł i zachowana odmowa starego K4

Przeczytano `AGENTS.md`, ADR, [propozycję dwóch UID](stage9_cross_uid_verifier_identity_proposal.md),
poniższe moduły i testy. Codebase Memory nie jest dostępne w katalogu narzędzi
ani jako lokalne CLI; brak wyniku grafu i `check_index_coverage`. Wnioski oparto
na bezpośredniej inspekcji, bez deklaracji kompletności indeksu.

| Źródło w bazie | Ustalenie istotne dla kontraktu |
| --- | --- |
| [`freshness_runtime_ipc.py`](../../bot_core/freshness_runtime_ipc.py), `_check_server`, `verify_exact_candidate` | Stary K4 nadal obowiązuje; nie ma alternatywy ani opcji pominięcia. |
| [`linux_service_ipc.py`](../../bot_core/linux_service_ipc.py), `capture_process_identity`, `PinnedReadOnlyPeer` | Wyłącznie `PermissionError` otwarcia `/proc/PID/exe`, przy `allow_missing_executable=True`, pozostawia `exe_sha256=None`. Inne błędy nie stają się pomiarem. |
| [`linux_physical_ipc.py`](../../deployment/linux_physical_ipc.py), `ExpectedPeerProcess`, `_pinned_verifier_socket` | Expected wymaga 64 znaków hex; chroniona ścieżka i socket-bound PIDFD nie zastępują hasha. `IPC_BLOCKERS` i `require_ipc_qualification` nadal blokują. |
| [`freshness_service_identity.py`](../../bot_core/freshness_service_identity.py), `_STATIC_PROPERTIES`, `_DYNAMIC_PROPERTIES`, kwalifikator | Jest dokładne sprawdzanie unitu i live exe; brak nowego dowodu ownera busa, boot ID, InvocationID i jawnego zakazu delegacji. Kwalifikacyjny `pidfd_open(MainPID)` nie jest `SO_PEERPIDFD`. |
| [`production_local_installer.py`](../../deployment/production_local_installer.py), `_manifest`, `_trusted_bootstrap`; [kontrakt runtime](stage9_linux_signed_python_runtime.md) | Podpisane dokładne bajty, osobny zaufany bootstrap i manifest aplikacji v2 wiążą runtime. Nie dowodzą ochrony startu/ciągłości K4-v2. |
| [Obecny unit](../../deployment/systemd/cryptohunter-freshness-verifier.service), bootstrap i [przegląd platformy](stage9_hybrid_platform_review.md) | `Type=simple`, `ExecStartPre`, chronione pliki i sandbox nie zapewniają seal przed listen ani zakazu późniejszego exec/fork/FD. Gotowość systemd nie oznacza gotowości K4-v2. |
| [Testy tożsamości](../../tests/security/test_freshness_runtime_ipc_identity.py), [transportu](../../tests/security/test_freshness_runtime_ipc.py), [usługi](../../tests/security/test_freshness_service_identity.py), [peer](../../tests/security/test_linux_service_ipc.py) i [deploymentu](../../tests/deployment/test_linux_physical_ipc.py) | Modele NSS/systemd/instalacji i własne AF_UNIX/PIDFD są regresjami; nie są natywnym dowodem odrębnych UID ani zatwierdzeniem aplikacji. |

Dokładna ścieżka odmowy: `PinnedReadOnlyPeer` zachowuje obserwowane `None`.
`_check_server` porównuje je z wymaganym `ExpectedPeerProcess.executable_sha256`
i rzuca `PermissionError("APPROVED_VERIFIER_PROCESS_REQUIRED")` przed `started=True`
i pierwszym `sendall`. `PermissionError` jest `OSError`; obsługa w
`verify_exact_candidate` zwraca **UNAVAILABLE / TRANSPORT_OR_IDENTITY_UNAVAILABLE**,
z zerem wysłanych bajtów aplikacyjnych. Wewnętrzny kod odmowy nie jest kodem
wyniku publicznego. Po rozpoczęciu send niejednoznaczność to **OUTCOME_UNKNOWN**,
bez retry. Żaden z tych warunków ani plików produkcyjnych nie jest zmieniany.

**Różnica względem K4:** obecny klient musi sam odczytać bieżący hash executable.
K4-v2 proponuje inną podstawę: niezależnie zatwierdzony start całego Python
closure, dokładna instancja systemowego managera, jądrowo związany kanał i
egzekwowana ciągłość. Nie jest zgodą na `None` w starym predykacie ani fallbackiem.
Hash na dysku dowodzi zatwierdzenia pliku; live hash interpretera także nie
dowodzi integralności pamięci Pythona. K1/K3/K5 i transportowe K8 pozostają;
K6/K7 wymagają osobnej implementacji i kwalifikacji nowej epoki oraz startu.

## Granice zaufania i zagrożenia

Zaufane: jądro, systemowy manager root, administrator instalacji, niezależny
autorytet zatwierdzający wydanie, bootstrap i zamknięcie zatwierdzonej aplikacji.
Klient musi otrzymywać expected/politykę niezależnie od badanego serwera.
Napastnik T1/T2/T3 może uruchamiać zwykły kod odpowiednio jako runtime, inny UID
lub **ten sam UID verifiera**, także `unconfined`; nie ma root/capabilities
ani praw zapisu zatwierdzonych materiałów. Dedykowane konto/nologin nie usuwa T3.

| Odrębna własność | Zakres proponowanej zgody i granica |
| --- | --- |
| Uwierzytelnienie instancji | IDENTITY_ACCEPT: ten kanał pochodzi od dokładnego zatwierdzonego startu i nie został oddany innemu odbiorcy. |
| Odporność na ingerencję lokalną | E07/E08 muszą wykluczyć T3/T5/T6 od narodzin procesu, nie dopiero po seal. Sygnały/DoS pozostają ryzykiem, jeśli osobna polityka ich nie ogranicza. |
| Podatność wewnątrz verifiera, T7 | RCE/parser/crypto/FFI może dać złą decyzję w prawidłowo uwierzytelnionej instancji. Python może wykonywać bytecode bez exec, nowego pliku i PROT_EXEC. Nie dowodzi się wykluczenia dowolnego RCE; wymóg takiej absolutnej gwarancji daje STOP/NOT_PROVEN dla Python i Rust. |
| Autoryzacja PostgreSQL | AUTHORITY_ACCEPT wymaga osobnej izolacji T3 i praw SQL, testów semantyki i trwałości. Peer OS UID nie uwierzytelnia kodu/cgroup. |

Koniunkcja przeciwdziała: podszyciu T1/T2, obcemu procesowi tego samego UID T3,
podmianie plików/unitu/środowiska/socketu T4, ingerencji w start T5, późniejszemu
exec/fork/delegacji FD T6 i użyciu starej instancji po restarcie. Każda obrona
jest warunkowa do czasu natywnego dowodu. Rezydualne pozostają T7, DoS, błędy
zatwierdzonego kodu/polityk i dostawców, podatności jądra/LSM/systemd, kompromitacja
zaufanego root T8 oraz odrębne lifecycle/replay/AccountGenesis/IPC_BLOCKERS.
Same zgodne snapshoty przed/po nie zamykają przerwy między nimi.

## Dokładny predykat i zamknięta lista dowodów przed pierwszym payload

Dla jednego połączenia `C`, zatwierdzonej instalacji `I` i epoki `J`:
**K4-v2(I,J,C) = E01 ∧ E02 ∧ E03 ∧ E04 ∧ E05 ∧ E06 ∧ E07 ∧ E08 ∧ E09.pre**.
Wynik odpowiedzi wolno przyjąć tylko z **E09.post** i tą samą epoką/kanałem.
Obowiązuje dokładna zgodność wszystkich pól, bez wyboru podzbioru/fallbacku.
Przed payload E09 zawiera także dowód, że kontrola po odpowiedzi jest wymagana
przez zatwierdzony klient TEST_ONLY; przyszłej obserwacji nie przedstawia się
jako już wykonanej. Brak/niedostępność dowodu: **BLOCKED/NOT_PROVEN i odmowa**;
sprzeczność lub obejście: **REJECT**. Nic nie jest wysyłane „dla sprawdzenia”.
Przed payload dopuszczalny jest sam connect i lokalna obserwacja, bez hello od
klienta; challenge/nonce i nagłówek ramki również są payload.

Każdy pakiet E ma identyfikator kontraktu, commit/build, zatwierdzenie instalacji,
źródło niezależne od serwera, czas monotoniczny obserwacji, boot/InvocationID,
zakres namespace i wynik kontroli spójności. Dowody statyczne/natywna kwalifikacja
nie wygasają czasowo w zastępstwie rechecku: są ważne tylko dla dokładnych bajtów,
wersji platformy i niezmienionej polityki. Dowód dynamiczny obowiązuje wyłącznie
w bieżącej wymianie, z uchwytami utrzymanymi do końca i istniejącym deadline.

### E01 — zatwierdzona instalacja i dokładne chronione bajty

Niezależne zatwierdzenie manifestu instalacji/aplikacji i runtime wiąże commit,
program, bootstrap, unit, polityki i pełne closure: interpreter, stdlib, wheels,
rozszerzenia, loader, OpenSSL/libpq i konfigurację ładowania. Odczyt podpisu
przed interpretacją, z chronionym zewnętrznym kluczem i bootstrapem. Niezapisywalne
przez T1–T3 pliki/przodkowie/mounty, zatwierdzone inode/ownership/mode/ACL i linki;
brak cwd/user-site/PYTHONPATH/.pth, pluginów i obcego bytecode. Zatwierdzenie
obecnych bajtów nie zatwierdza przyszłego launchera; żaden pin nie jest odświeżany.

### E02 — rzeczywisty systemowy manager

Stały lokalny system bus z chronioną ścieżką i jawnym, zatwierdzonym adresem,
bez adresu z env/użytkownika. Unikalny owner `org.freedesktop.systemd1`, jego
kernelowe credentials/PID/UID i niezależna korelacja z systemowym PID 1, bootem,
namespace oraz zatwierdzonym systemd. Chronione D-Bus/Polkit wykluczają kontrolę
jednostki przez T1–T3. User manager, podstawiony bus/proxy, zmiana ownera i brak
dostępu odmawiają. Zaufanie do root/jądra jest jawne; odpowiedź serwera aplikacji
z metadanymi systemd nie jest tą obserwacją.

### E03 — dokładna efektywna jednostka

Zatwierdzone materializowane bajty i odczyt efektywnego unitu: fragment, lista
drop-inów (dla pierwszego profilu pusta), `ExecStartPre`/`ExecStart`, argv,
environment oraz źródła env, working/root directory, namespaces, sandbox,
capabilities, grupy, polityki startu i reload. Brak niezatwierdzonych transient
properties, env managera/generatorów, wrapperów, delegacji cgroup (`Delegate=no`),
socket activation i dodatkowych procesów. Bez `NeedDaemonReload`; restart lub
zmiana konfiguracji unieważnia expected. Unit `active/running` nie dowodzi seal.

### E04 — jedna instancja i epoka

Rzeczywiste różne niezerowe UID runtime/verifiera; wszystkie real/effective/
saved/fs UID/GID i zamknięte grupy zgodne z instalacją. Verifier bez dodatkowych
grup poza dopuszczoną grupą IPC. `MainPID=ExecMainPID>0`, dokładny niedelegowany
cgroup v2 i brak niezatwierdzonych członków. Dodatnie stat22 start ticks, boot ID
i niepuste poprawne InvocationID z managera tworzą epokę z instalacją i PID.
Korelacja czasu systemd i ticków uwzględnia jednostki/HZ, nie równość surowych liczb.
Wspólny zaufany widok PID/user/mount namespace; nieczytelne pola to odmowa.

### E05 — związanie z tym samym połączeniem

AF_UNIX/SOCK_STREAM: `SO_PEERCRED` i **socket-bound SO_PEERPIDFD** pobrane z
tego samego połączonego deskryptora, zgodne z E04. PID w fdinfo i żywotność
sprawdzone przed/po obserwacji, uchwyt zachowany; brak ABI/wsparcia to odmowa.
`pidfd_open(PID)` nie jest zamiennikiem. Poświadczenia stream mogą pochodzić z
listen i nie zmieniają się po delegacji FD; dopiero E07/E08 wykluczają obcego
handlera. Nie ma dziedziczonego listenera ani socket activation.

### E06 — chroniony endpoint AF_UNIX

Stała zatwierdzona ścieżka w zweryfikowanym mount namespace. Utrzymane dirfd
przodków i inode socketu, bez niezatwierdzonych symlinków/ACL/hardlinków/mountów.
Zachowane wymagania obecnego K5: wyżsi przodkowie root, bez zapisu grupy/innych;
bezpośredni katalog verifier: UID verifiera, GID IPC, bez zapisu grupy/innych;
socket: UID verifiera, GID IPC, `0660`, jeden link. Recheck dev/inode/owner/mode/
ctime i wszystkich uchwytów. Same-UID może próbować podmiany katalogu/socketu:
ścieżka sama nie wystarcza, konieczne E04/E05/E08 i recheck przed payload.

### E07 — ochrona startu od narodzin procesu

Zatwierdzona polityka chroni od systemowego managera przez zmianę credentials,
exec i loader do seal przed T3/T5: attach/seize/TRACEME, process_vm, proc-mem,
przejęcie FD, wcześniejszego tracera, dzieci, współdzielonych mapowań i kanałów.
Zaufany rodzic i zamknięty inwentarz dziedziczenia, bez uprawnień napastnika
CAP_SYS_PTRACE w userns celu. Natywna kwalifikacja musi odpowiadać dokładnej
polityce i kolejności startu. Późne dumpable=0/TracerPid=0, dedykowany UID lub
profil samej ofiary nie dowodzą tej historii. Wybór polityki: D03, poniżej.

### E08 — ciągłość przed udostępnieniem i podczas obsługi

Przejrzany zatwierdzony kod startu usuwa obce FD/dzieci/ringi/mapowania, ładuje
zamknięcie, ustawia nieodwracalnie NNP/dumpable i egzekwowaną politykę **przed
bind/listen oraz przed odczytem danych**. Niepowodzenie zatrzymuje start bez
endpointu. Zakaz późniejszych execve/execveat (także ten sam Python/FD),
fork/vfork/clone/clone3, delegacji/pozyskania FD (SCM_RIGHTS w obie strony,
sendmsg/sendmmsg/recvmsg/recvmmsg, pidfd_getfd), io_uring i dróg cofania ochrony
(credentials, dumpable, PR_SET_PTRACER, namespaces, profil). Wszystkie wątki i ABI/x32 objęte;
brak alternatywnej drogi syscall lub wcześniejszej kopii FD. To wymagane własności,
nie nowa allowlista seccomp. Dokładny mechanizm musi przejść review i N10–N13
na VM; `Seccomp: 2`/liczba filtrów nie dowodzą treści ani kolejności. Zaufany
start i niezależnie od aplikacji sprawdzona konfiguracja oraz kwalifikacja są
podstawą dowodową; serwer nie poświadcza własnego seal. Python/OpenSSL dziś
używa subprocess podczas bootstrapu, a aplikacja PG przed listenerem: konieczny
jest dowód zgodności rzeczywistej kolejności, nie przeniesienie modelu sondy.

### E09 — ponowne kontrole i wygaśnięcie

Bezpośrednio przed pierwszym send ponowne sprawdzenie E02–E06, ważności E01/E07/E08,
własnej tożsamości runtime i tej samej epoki/kanału. Po odpowiedzi to samo,
przed przyjęciem wyniku. Śmierć, restart, boot, zmiana ownera managera, unitu,
polityki lub materiałów cofa admission; nowa instancja wymaga pełnego pakietu.
Nie używać cache z poprzedniej wymiany. Jeden request/connection, istniejący
deadline **3,0 s dla całej wymiany** (`EXCHANGE_TIMEOUT_SECONDS`), brak reconnect/retry.
Limit ramki aplikacyjnej pozostaje **4 MiB**. Przed send odmowa: UNAVAILABLE i 0 payload;
po rozpoczęciu send niepewność: OUTCOME_UNKNOWN, bez twierdzenia o rollbacku.
Ponowna zgodność nie przywraca ujawnionych danych; ciągłość egzekwują E07/E08.

## T5: decyzja o ochronie startu i wpływie na host

**Globalne Yama `ptrace_scope=2` nie jest logicznym warunkiem każdej realizacji
K4-v2. Jest wymaganą przesłanką profilu startu YAMA2, jeśli właściciel go wybierze.**
Profil wymaga Yama aktywnego w dokładnym kernelu, udowodnionej aktywacji wartości
2 w zaufanej fazie bootu **przed narodzinami pierwszej objętej instancji** i jej
nieprzerwanego utrzymania, braku wcześniejszego tracera oraz braku CAP_SYS_PTRACE
u napastnika w userns celu. Zapis „od bootu” z ADR wymaga takiego doprecyzowania
w D03: standardowe Yama zaczyna od 1, a sysctl.d może ustawić 2 w trakcie bootu.
Nie wymaga się tu milcząco zmodyfikowanego kernela z domyślną wartością 2;
właściciel musi jawnie zaakceptować dokładną kolejność aktywacji i dowód braku
wcześniejszej ingerencji. Odczyt sysctl po seal nie wystarcza. Zwykły exec może
ustawić dumpable=1 przed pierwszą instrukcją launchera; późne prctl nie zamyka
tego okna. Yama2 ogranicza ATTACH/TRACEME, process_vm i pidfd_getfd, lecz nie
wszystkie kontrole READ_FSCREDS (np. proc/root/cwd) ani sygnały/DoS.

Wpływ jest hostowy: zwykli użytkownicy tracą możliwość ptrace attach i śledzenia
dzieci/TRACEME, także dla gdb/strace; diagnostyka, profiler i zbieranie crashy
mogą wymagać uprzywilejowanego, osobno ocenionego dostępu. Trzeba wykazać wpływ
na konkretne narzędzia hosta, tryb odzyskiwania i brak poszerzenia capabilities
usług. Nie utożsamiać tego z zakazem wszystkich profilerów lub core dumps.

Alternatywy do osobnej decyzji i takiej samej macierzy dowodowej:

- Yama1 z wykazaną zaufaną relacją rodzica od startu, bez wyjątków PR_SET_PTRACER,
  wcześniejszego tracera i drogi zmiany relacji: **NOT_PROVEN**; sama wartość 1
  nie dowodzi tych warunków.
- Root-owned execute-only natywny ELF z niezależnie wymuszonym `fs.suid_dumpable`
  0/2 i poprawnym startem root-managera: kandydat źródłowy **NOT_PROVEN**, nie
  właściwość obecnego Pythona; wymaga oceny loadera, DAC/ACL i całego closure.
- LSM rzeczywiście mediujący także napastnika unconfined od narodzin procesu
  albo zmiana architektury startu: **NOT_PROVEN** na wskazanej platformie;
  AppArmor wyłącznie na ofierze Noble nie daje takiego dowodu. Nowy uprzywilejowany
  komponent zwiększa TCB i wymaga osobnego uzasadnienia/zgody.

Nie wybiera się alternatywy milcząco po nieudanej próbie. Brak zaakceptowanego
profilu i natywnego dowodu E07 utrzymuje BLOCKED. **Ten PR nie zmienia sysctl,
LSM, Polkit ani jakiejkolwiek polityki hosta.**

## T3: izolacja PostgreSQL przed AUTHORITY_ACCEPT

Obecne [pg_ident](../../deployment/postgresql/pg_ident.conf) i
[pg_hba](../../deployment/postgresql/pg_hba.conf) mapują tożsamość OS przez peer.
`PostgreSQLVerifierConnection.connect` w
[`freshness_semantic_verifier.py`](../../bot_core/freshness_semantic_verifier.py)
sprawdza lokalność i rolę sesji, nie kod/cgroup. Obcy proces UID verifiera,
który dosięga tego endpointu, może dostać rolę `freshness_crypto_verifier`.
**Publiczny dla tego UID socket PG z samym peer oznacza AUTHORITY_REJECT**,
nawet przy IDENTITY_ACCEPT.

Przed kwalifikacją autorytetu osobny zatwierdzony projekt musi zapewnić prywatny
endpoint PG za root-owned przodkiem, osiągalny wyłącznie dla autoryzowanych
instancji/namespace, bez TCP i alternatywnego publicznego socketu/credentiali
dających te prawa. Same mode/grupa nie odróżniają obcego procesu tego samego UID.
Izolacja obejmuje cały start PG/verifiera, restarty i runtime: `/proc/PID/root`,
cwd, fd, mountinfo/namespace, setns, dziedziczenie i przekazywanie FD, zachowane
połączenia/dirfd, podmiany mountów oraz dostęp do roli przez inne endpointy.
Yama2 nie rozstrzyga wszystkich dróg READ_FSCREDS. Ustawienie dumpable po otwarciu
połączenia PG nie dowodzi braku wcześniejszego wycieku.

N16 wymaga osobnego obcego procesu tego samego UID spoza jednostki i rzeczywistego
PG: poprawna instancja uzyskuje swoją rolę; obcy nie dostaje uprzywilejowanej
sesji przez żadną drogę. N18 niezależnie potwierdza, że runtime nie prepare,
verifier nie CAS i złe dane nie zmieniają stanu. Nie załatwia się T3 zmianą ról,
HBA lub SQL w tym PR. Pierwszy prototyp tożsamości **nie łączy PostgreSQL**.

## Natywna próba i niezależne oracle

Dopiero po ACCEPT etapu 1 i osobnej zgodzie na konkretną jednorazową VM:
Ubuntu 24.04 z systemd 255 jako rzeczywistym systemowym PID 1, dokładny pakiet
jądra/ABI (bez założenia, że każde GA/HWE jest równoważne), LSM/enforce,
Yama/sysctl, D-Bus/Polkit, mount/proc/cgroup namespaces i wersje narzędzi zapisane
w artefaktach. Rzeczywiste różne UID/GID, osobny obcy proces tego samego UID,
chronione materiały TEST_ONLY i cały inwentarz E01–E09. Bez sekretów produkcji.
Pierwszy endpoint wykorzystuje zatwierdzony start dla kandydata Python i tylko
syntetyczny challenge/odpowiedź; parser dokumentów, crypto/PG/PREPARED to później
N16–N18. Nie wystarczy sonda, która chroni jedynie własny proces.
Zachowane są osobne bramki [harnessu VM](../runbooks/STAGE9_LINUX_VM_HARNESS.md)
i późniejszej [kwalifikacji usługi](../runbooks/STAGE9_FRESHNESS_SERVICE_VM_ACCEPTANCE.md);
ten kontrakt nie udziela zgody operatorskiej na żadną z tych prób.

Każdy podprzypadek tabeli jest obowiązkowy i dostaje oddzielny rekord:
ID, operacja/aktor/faza, oczekiwanie, rzeczywisty wynik/errno/sygnał, źródło
odmowy, kontrola osiągalności, niezależny oracle, liczniki payload i epoka.
Brak implementacji/wspieranego ABI/praw do obserwacji to BLOCKED/NOT_PROVEN,
nie SKIP/PASS. Wspólne zasady oracle:

- **O-BYTES:** niezależny kontroler próby, poza badanym procesem i jego UID,
  obserwuje syscalle transportu/liczniki obu końców, PID odbiorcy i granicę
  pierwszego odczytu. Osobno bajty wysłane przez poprawnego klienta, odczytane
  i przetworzone przez serwer. Sam log/counter serwera nie wystarcza. Bajty
  wysłane przez napastnika do kolejki nie dowodzą odczytu przez serwer.
- **O-KERNEL:** kontrola dodatnia dowodzi, że operacja dochodzi do właściwej
  ścieżki na tym kernelu/ABI z właściwymi argumentami i credentials. Zewnętrzna
  obserwacja/audyt rozstrzyga mechanizm odmowy; porównanie kontrolne zmienia
  wyłącznie badaną ochronę na osobnym wariancie zatwierdzonej VM. Nie osłabia
  badanej instancji. EPERM/EINVAL, ENOSYS, timeout i odmowa przez inną ochronę
  bez identyfikacji nie zaliczają testu. Brak możliwości atrybucji to NOT_PROVEN.
- **O-STATE:** niezależny od klienta/verifiera kontroler odczytuje stan managera,
  uchwyty proc/socket/mount i materiały; dla PG osobne połączenie kontrolne i
  rzeczywisty log/snapshot transakcyjny. Nie ufa raportowi aplikacji o własnej
  tożsamości, seal ani efektach SQL. Sposób obserwacji nie może umożliwiać
  napastnikowi nowych praw ani zmieniać sprawdzanej ochrony.

| ID | Obowiązkowe przypadki | Niezależny oracle i warunek zaliczenia |
| --- | --- | --- |
| N01 | Niezmieniony klient, rzeczywiste dwa UID, `exe_sha256=None`, pozostałe warunki spełnione. | O-STATE potwierdza PermissionError exe i dojście do `_check_server`; O-BYTES: 0 payload; wynik UNAVAILABLE. Odmowa wcześniejszej bramki nie zalicza K4. Istniejąca modelowa regresja jest osobnym testem kodu. |
| N02 | Właściwy Python endpoint, komplet E01–E09, dokładny manager/epoka/kanał. | O-STATE koreluje niezależne expected z kernelem; O-BYTES: dokładnie jedna wymiana syntetycznego challenge z jedną zgodną odpowiedzią i tym samym PID odbiorcy. Brak wstrzykiwania expected do observed. |
| N03 | Zły UID klienta/serwera; obcy proces UID verifiera poza jednostką, także identyczne zatwierdzone bajty. | O-STATE potwierdza rzeczywiste credentials/cgroup; O-BYTES: poprawny klient do złego serwera wysyła 0; serwer dla złego klienta czyta/przetwarza 0. Pozytywna N02 wyklucza martwy endpoint. |
| N04 | Każda klasa E01: zmienione bajty/runtime/import/cache/klucz, brak/podpis/inwentarz, ownership/mode/ACL/mount. | Kontroler niezależnie porównuje mutację z zatwierdzonym materiałem i obserwuje odmowę instalacji/startu/admission przed payload; O-BYTES 0. Zgodne materiały przechodzą N02. |
| N05 | Fałszywy system/user bus, env adresu, inny owner/PID/UID managera, utrata busa i brak prawa odczytu. | O-STATE potwierdza endpoint i credentials ownera; każda mutacja E02 odmawia przed send, O-BYTES 0. Sam błąd połączenia bez zidentyfikowanej przyczyny nie zalicza podszycia. |
| N06 | Drop-in, transient override, argv/ExecStartPre, env/file/manager env, delegacja, socket activation, reload. | Niezależny odczyt efektywnego unitu wykazuje zmieniony element; brak admission i 0 payload. Powrót do materiałów N02 przy nowej epoce stanowi kontrolę dodatnią. |
| N07 | Każde pole E04/E05 osobno: UID/GID/grupy, MainPID/ExecMainPID, cgroup, ticks, boot, InvocationID, credentials/pidfd. | Kontroler zachowuje rzeczywiste różniące się krotki/uchwyty, odmawiający klient wysyła 0. Nie wystarczy mock; pole niewymuszalne natywnie pozostaje NOT_PROVEN. |
| N08 | Podmiana socketu/katalogu, inode/owner/mode/ACL/symlink/mount, także wyścig przed send i po odpowiedzi. | O-STATE przez zachowane uchwyty potwierdza konkretny drift; O-BYTES przed send 0; po send wynik OUTCOME_UNKNOWN i brak retry. Brak podmiany i martwego serwera nie zalicza próby. |
| N09 | Brak/niedostępność każdego wymaganego w danej fazie pola E01–E09; brak SO_PEERPIDFD/procfs/systemd; server hash, ready, `Seccomp: 2`, `pidfd_open` i expected jako substytut. | Niezależne odcięcie wskazanego źródła lub podanie substytutu, pozostałe warunki dostępne: przed send brak admission/0 payload/UNAVAILABLE; utrata źródła lub E09.post po rozpoczęciu send daje OUTCOME_UNKNOWN bez retry. Bez fallbacku. Kontrola pokazuje rzeczywisty brak, nie pusty mock raportu. |
| N10 | T5 przed pierwszą instrukcją/loaderem i podczas startu: attach/seize/TRACEME, process_vm_readv/writev, proc-mem, pidfd_getfd, READ_FSCREDS/root/cwd/fd. | O-KERNEL z uzgodnionymi punktami synchronizacji i źródłami kernela: każdy zabroniony dostęp odrzucony przez wybrany mechanizm lub start odrzucony przed ready. Skończona seria wyścigów bez naruszenia sama nie dowodzi ochrony od narodzin. |
| N11 | Wcześniejszy tracer/dziecko, odziedziczony listener/FD/ring/mapowanie; błąd NNP/dumpable/instalacji polityki/wątku/ABI; próba gotowości przed seal. | O-STATE niezależnie wykazuje wprowadzone skażenie i kolejność; O-KERNEL/O-BYTES potwierdzają zamknięcie startu przed bind/listen i odczytem. Gotowy plik lub końcowe TracerPid=0 nie zaliczają. |
| N12 | Po seal execve/execveat: inny program, ten sam Python/inny kod, AT_EMPTY_PATH; fork/vfork/clone/clone3; zmiana credentials/dumpable/PR_SET_PTRACER/namespace/profilu. | O-KERNEL dla każdej operacji i wszystkich dopuszczonych ABI; brak zmiany wykonawcy/dziecka/cofnięcia ochrony potwierdzony O-STATE. Brak dostępnej instrukcji/syscalla nie staje się PASS. |
| N13 | SCM_RIGHTS listenera i połączenia w obie strony, sendmsg/sendmmsg/recvmsg/recvmmsg, pidfd_getfd z procesu T3, io_uring_setup/enter/register, wcześniejsze ringi i obcy ABI/x32. | O-KERNEL przypisuje odmowę; niezależny współpracujący odbiorca nie uzyskuje FD/payload, O-BYTES identyfikuje wykonawcę. Zgodne stare peer credentials i żywy rodzic nie zaliczają delegacji. |
| N14 | Śmierć/restart/nowy boot lub InvocationID; drift managera/polityki przed send, w trakcie i po odpowiedzi; timeout/partial send. | O-STATE potwierdza epokę i moment awarii; O-BYTES: przed send 0/UNAVAILABLE, po rozpoczęciu send OUTCOME_UNKNOWN, brak ponownego requestu. Stare expected nigdy nie przyjęte po restarcie. |
| N15 | Jedna ramka/wymiana i deadline, nadmiarowe/ucięte/zbyt duże dane, zła odpowiedź challenge, awaria kanału. | Zewnętrzny zegar i O-BYTES oraz wcześniej zamrożone wektory syntetycznego protokołu: właściwy wynik albo odmowa/niejednoznaczność, bez retry i akceptacji wyniku bez E09.post. Nie używać parsera badanego endpointu jako jedynego oracle. |
| N16 | T3/PG: obcy proces UID verifiera poza jednostką i inni UID; wszystkie ścieżki endpointu z sekcji T3, podczas startu, po seal i restarcie. | O-KERNEL/O-STATE plus log i niezależne połączenie PG: właściwa instancja dostaje dokładną rolę, obca żadnej uprzywilejowanej sesji. Próby dochodzą do prawdziwego endpointu/kontroli dostępu; brak uruchomionego PG nie zalicza. |
| N17 | Pełne obecne K8/P7: trzy obiekty, framing !I/JCS/base64url/4 MiB, Ed25519/domeny/selektory i binding PREPARED; zły podpis, Unicode/duplikaty/bool/int/zakresy. | Niezależnie uzgodnione wektory z dokładnymi bajtami/wynikami i odczyt PG: zero rozbieżności, odrzucone dane nie tworzą przygotowania. Istniejący Python nie jest jedynym oracle własnych wyników. |
| N18 | Runtime bez prepare, verifier bez CAS, obcy UID bez praw; błędne dane bez efektów; awaria/timeout COMMIT i trwałość po restarcie testowego PG. | Osobna sesja kontrolna/katalog praw i stan przed/po, dziennik PG, dokładny identyfikator kandydata: tylko dozwolony efekt i trwałość. Nierozstrzygnięty COMMIT daje OUTCOME_UNKNOWN/brak retry, nigdy zapewnienie rollbacku. |

Nie publikować sekretów lub payload produkcji. Zachować źródła/build/inwentarz,
zgodę VM, niezmienny korpus, surowe dowody wszystkich podprzypadków, wersje
platformy, oczekiwania i cleanup/snapshot. Modele/testy statyczne oznaczać MODELED;
nie stają się wynikiem tej macierzy. Nie wykonano tu prób administracyjnych,
instalacji, zmian hosta ani kwalifikacji fizycznego Linuxa.

## Reguły decyzji i jawne zatwierdzenia

**ACCEPT etapu 1** wymaga osobnego rekordu właściciela i niezależnego recenzenta:
wersja/commit kontraktu, rozstrzygnięcia D01–D08, zamrożony korpus i szczegółowe
oczekiwania podprzypadków, źródła oracle, mierzalne budżety i zakres kolejnego
PR. Brak decyzji daje STOP dla implementacji, nie domniemaną zgodę.

| ID | Decyzja wymagana od właściciela; obecnie wszystkie PENDING |
| --- | --- |
| D01 | Jawna akceptacja alternatywnej podstawy K4-v2 tylko TEST_ONLY, koniunkcji E01–E09, wersjonowania i pozostawienia starego guardu. |
| D02 | Zaufanie closure istniejącego Pythona i zaakceptowanie rezydualnego T7 bez obietnicy odporności na dowolne RCE; Rust tylko po osobnym uzasadnieniu. |
| D03 | Dokładny profil ochrony startu: YAMA2 z wpływem na cały host albo osobno przejrzana alternatywa; udowodniona aktywacja podczas bootu przed narodzinami instancji, dalsza niezmienność, capabilities, aktualizacja/recovery i READ_FSCREDS. Jawne doprecyzowanie wymogu ADR „od bootu”. |
| D04 | Rozdzielenie tożsamości od autorytetu; zatwierdzony zakres/projekt T3/PG albo jawny AUTHORITY_REJECT przy publicznym peer-only PG. Pierwszy prototyp bez PG. |
| D05 | Lifecycle i cofanie admission po restarcie/boot/drifcie; brak cache/retry i sposób zachowania OUTCOME_UNKNOWN. |
| D06 | Niezmienny korpus N01–N18 i osobne wyniki każdego podprzypadku, niezależne oracle/atrybucja, zero dopuszczonych pominięć lub różnic. |
| D07 | Liczbowe budżety czasu/pamięci/CPU/startu i liczba powtórzeń, zakres kompatybilności platformy, bez zwiększania obecnych deadline/progów CI. Brak wartości blokuje etap 1. |
| D08 | Właściciel i niezależny recenzent decyzji, osobna zgoda na wskazaną jednorazową VM/jej polityki/cleanup oraz zakres nowego PR implementacyjnego dopiero po ACCEPT etapu 1. |

**IDENTITY_ACCEPT (wyłącznie zakres VM):** wszystkie E01–E09 i N01–N15
spełnione natywnie bez pominięcia obowiązkowego podprzypadku, prawdziwe dwa UID,
N02 dodatni, negatywy z właściwymi licznikami, przypisane mechanizmy odmowy,
brak obcego handlera i wycieku przed admission. Wymagana niezależna decyzja
po przeglądzie surowych dowodów, nie sam napis PASS od harnessu. Nie wymaga
jeszcze N16–N18; ich wynik pozostaje NOT_RUN, autorytet BLOCKED/REJECT.

**AUTHORITY_ACCEPT (późniejszy zakres VM):** dodatkowo N16–N18, pełne K8,
zatwierdzona izolacja T3/PG i zgodność budżetów. Nie zwalnia z pozostałych bramek
produktu i nie aktywuje K4-v2 w produkcji.

**REJECT:** fałszywe uwierzytelnienie, payload poprawnego klienta do odrzuconego
serwera lub odczyt/przetworzenie danych złego klienta, stara epoka, dopuszczona
delegacja/niechroniony start, dowód sprzeczny, bypass, substytut zamiast obserwacji,
niezatwierdzone rozszerzenie TCB. Dostęp T3 do uprzywilejowanego PG daje
**AUTHORITY_REJECT**; sam nie unieważnia niezależnie dowiedzionej tożsamości.
Zatrzymać próbę danego zakresu i zachować dowody. **NOT_PROVEN:** brak/niedostępność materiału, niewykonany
test, brak obsługi platformy, brak kontroli dodatniej lub nieustalona przyczyna
odmowy. Daje operacyjne BLOCKED/odmowę przed payload; nie wolno oznaczać ACCEPT.
REJECT ma pierwszeństwo w swoim zakresie, jeśli istnieje wykazane naruszenie; brak innych dowodów
nie degraduje go do NOT_PROVEN. Sprzeczne wymagania/absolutne wykluczenie RCE
zatrzymują etap 1. Modele i odmowy przez niezidentyfikowaną ochronę nigdy nie
kwalifikują natywnego testu.

## Zakres walidacji tego pakietu

Test schematu/inwentarza i względnych odnośników sprawdza kompletność dokumentu,
nie działa jako klient ani kwalifikator. Regresje obecnej odmowy, kontraktów
instalacji, tożsamości i bramki AccountGenesis zachowują dotychczasowy zakres.
Wyniki poleceń i stan CI dla końcowego HEAD należą do opisu Draft PR:

```bash
git diff --check
python -m pre_commit run ruff-format --all-files
python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
python -m pytest -q tests/architecture/test_stage9_verifier_identity_acceptance_contract.py tests/security/test_freshness_runtime_ipc.py tests/security/test_freshness_runtime_ipc_identity.py tests/security/test_freshness_service_identity.py tests/security/test_linux_service_ipc.py tests/deployment/test_linux_physical_ipc.py tests/deployment/test_production_local_installer.py tests/architecture/test_m05_account_genesis_freshness_authority_production_local_authentication_gate.py
python scripts/quality/betterleaks_hook.py docs/architecture/stage9_verifier_identity_acceptance_contract_v1.md tests/architecture/test_stage9_verifier_identity_acceptance_contract.py
```

Bez zmian verifiera/klienta/AccountGenesis, PostgreSQL/ról/uprawnień,
podpisów/digestów/pinów/manifestu, `deployment/stage9_current_status.json`,
CI/progów/bramek. Brak opcji pominięcia K4. Osobny Draft PR, bez merge;
queued/pending CI nie oznacza PASS i nie uzasadnia restartu jobów.
**Wynik: kontrakt do decyzji; K4-v2 niezatwierdzony i nieaktywny,
Stage 9 pozostaje IN_PROGRESS / PRODUCTION_NO_GO.**
