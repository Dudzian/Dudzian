# ADR: tożsamość FreshnessVerifier między rzeczywiście różnymi UID

Data: 5 listopada 2026 r. Repozytorium: `Dudzian/Dudzian`.
Baza sprawdzona: `main` = `e0215b39fd447b548c7b660bfda90d04492cae91`
(merge PR #3253). Status decyzji: **PROPOZYCJA DO ZATWIERDZENIA**.
**Stage 9: IN_PROGRESS / PRODUCTION_NO_GO; STAGE9_PRODUCTION_READY=NO.**

## Decyzja przed następnym eksperymentem

Oddzielamy **mechanizm tożsamości** od **technologii verifiera**. Proponowany
mechanizm to zaufany start, systemowy manager, jądrowe związanie kanału z dokładną
instancją i egzekwowana ciągłość; wymaga takiego samego dowodu dla A (Python)
i B (natywny Rust). **Warunkowo rekomendujemy zbadanie B**, bez zgody na port
produkcji: mniejsze domknięcie zależności i powierzchnia wykonania są hipotezami,
a nie wykazaną przewagą nad A. Wybór określają wspólne kryteria poniżej.
**Fallback mechanizmu: C**, wąski obserwator, jeżeli wymagany pozostanie
niezależny bieżący pomiar obrazu. C może obsługiwać A albo B; nie rozstrzyga
technologii i nadal wymaga ochrony ciągłości. Sam hash ELF, Rust lub raport
PID/hash nie ustanawiają tożsamości.

Żaden wariant nie spełnia obecnego kontraktu bez osobnej zmiany jego podstawy
dowodowej. Ten ADR nie zatwierdza tej zmiany. Nie implementować nowego klienta,
launchera ani kolejnej sondy, dopóki etap 1 nie uzyska jawnego ACCEPT.
Pierwszy cel po tej decyzji to poprawne uwierzytelnienie TEST_ONLY z dwóch UID,
a nie deklaracja gotowości produktu.

Gwarancję formułujemy jako **pochodzenie instancji i kanału od zatwierdzonego
uruchomienia**, z ochroną przed ingerencją innych zwykłych procesów, także tego
samego UID. Nie formułujemy gwarancji, że dowolna podatność samego zaufanego
verifiera nie pozwoli wykonać niezaufanego kodu. Ta ostatnia własność jest
odrębnym wymaganiem: jeżeli pozostanie absolutnym warunkiem ACCEPT, **STOP:
żaden z A/B/C nie ma dowodu jego spełnienia**. Nie wyłącza się jej milcząco.

## Co sprawdzono i co pozostaje źródłem prawdy

Przeczytano `AGENTS.md`, opisy, rozmowy/reviews i kompletne listy plików PR
[#3248](https://github.com/Dudzian/Dudzian/pull/3248),
[#3251](https://github.com/Dudzian/Dudzian/pull/3251),
[#3252](https://github.com/Dudzian/Dudzian/pull/3252),
[#3253](https://github.com/Dudzian/Dudzian/pull/3253), a następnie odpowiadające
im źródła pod powyższą bazą. Dostępne narzędzia nie udostępniają Codebase Memory;
zapytanie o serwer `codebase-memory` zwróciło `unknown MCP server`.
Nie ma wyniku `check_index_coverage` ani deklaracji kompletności indeksu.
Wnioski poniżej opierają się na bezpośrednich odczytach kontraktów, kodu i testów.

| PR i źródło w repozytorium | Wniosek istotny dla decyzji | Granica dowodu |
| --- | --- | --- |
| #3248: [kontrakt klienta](stage9_linux_freshness_runtime_ipc.md) | Istnieje ścisły jednorazowy transport; brak dostępnego executable obcego UID nadal blokuje send. | Rootless AF_UNIX/PIDFD i modele instalacji nie kwalifikują dwóch UID, systemd ani PostgreSQL. |
| #3251: [przegląd dwóch UID](../audits/stage9_cross_uid_identity_review_2026-11-05.md), [runbook](../runbooks/stage9_cross_uid_native_identity.md) | Harness i publikacja dowodów mają własne bramki; zgodność formatu nie jest NATIVE PASS. | Próba właściwej usługi NOT_RUN; oczekiwana odmowa z zerem bajtów nie jest dodatnim dowodem uwierzytelnienia. |
| #3252: [model seal](stage9_test_only_verifier_seal.md), [raport](../audits/stage9_verifier_seal_test_only_2026-11-05.md) | Model późnego seal nie dowodzi ochrony wcześniejszego startu, Python w pamięci ani braku delegacji FD. | cBPF jako dane, MODELED/NOT_RUN; REJECT pełnej gwarancji tego projektu. |
| #3253: [przegląd platformy](stage9_hybrid_platform_review.md), [hybryda](stage9_hybrid_verifier_test_only.md), [raport](../audits/stage9_hybrid_test_only_2026-11-05.md) | W badanym Ubuntu GA AppArmor profilu ofiary pomija mediację ptrace wobec unconfined tracera. Exec może przywrócić dumpable przed kodem launchera. Python wykonuje bytecode bez nowego exec/pliku/PROT_EXEC. | DAC/Yama mogą niezależnie odmówić; nie wykazano wykonanego exploitu. Sonda dowodzi tylko własnych odmów seccomp; pełna macierz Noble/systemd255/FD NOT_RUN. |

Historyczny REJECT nie zakazuje badania innego jawnego modelu zagrożeń.
Scalenie tych PR nie jest formalną akceptacją nowej podstawy tożsamości.
Historyczne liczby testów nie są wynikami walidacji tego ADR.

## Obecne warunki akceptacji i dokładna blokada

Odnośniki dotyczą źródeł w sprawdzonej bazie. Nazwy symboli ułatwiają
odtworzenie przeglądu, gdy numery linii zmienią się w przyszłości.

| ID | Obecny warunek | Źródło i test egzekwujący |
| --- | --- | --- |
| K1 | Niezerowe, różne UID runtime/verifiera; dokładne real/effective/saved UID/GID runtime; zamknięte grupy, właściwa grupa IPC. | [`_require_runtime_identity`](../../bot_core/freshness_runtime_ipc.py), linie 78–99; `tests/security/test_freshness_runtime_ipc_identity.py`, testy NSS/UID/GID/grup. |
| K2 | `ExpectedPeerProcess` pochodzi z niezależnie zatwierdzonej obserwacji wydania i systemd; odpowiedź serwera ani caller-selected PID nie ustanawiają zaufania. | [`ExpectedPeerProcess`](../../deployment/linux_physical_ipc.py), linie 57–74; `RuntimeFreshnessVerifierClient.__init__`, linie 281–293; `tests/deployment/test_linux_physical_ipc.py`. |
| K3 | Rzeczywisty AF_UNIX stream, `SO_PEERCRED`, **socket-bound** `SO_PEERPIDFD`, stabilny żywy proces. Późniejsze `pidfd_open(PID)` nie zastępuje związania z socketem. | [`PinnedReadOnlyPeer`](../../bot_core/linux_service_ipc.py), linie 147–204; `tests/security/test_linux_service_ipc.py`. |
| K4 | Dokładne PID, UID/GID, start ticks, **dostępny bieżący `exe_sha256` równy oczekiwanemu**, cgroup usługi i brak dodatkowych grup. | [`_check_server`](../../bot_core/freshness_runtime_ipc.py), linie 102–115; `test_model_incomplete_or_unisolated_server_identity_is_denied` w `test_freshness_runtime_ipc_identity.py`, linie 220–234. |
| K5 | Chronieni przodkowie i inode socketu, brak niedopuszczonych symlinków/ACL, dokładne ownership/mode; przypięcie i recheck. | [`_pinned_verifier_socket`](../../deployment/linux_physical_ipc.py), linie 350–431; testy podmian ścieżki w `test_freshness_runtime_ipc.py`. |
| K6 | K1/K3/K4/K5 przed send i po odpowiedzi; stabilne cmdline. | `verify_exact_candidate`, linie 331–367; testy driftu przed/po wymianie w `test_freshness_runtime_ipc.py`. |
| K7 | Podpisany runtime/aplikacja, dokładny skuteczny unit i bootstrap, chroniony program/inode, MainPID=ExecMainPID, PIDFD i argv. | [`freshness_service_identity.py`](../../bot_core/freshness_service_identity.py): `_STATIC_PROPERTIES`, `_exec_start`, `_bootstrap_start`, `_program_observation`; [podpisany runtime](stage9_linux_signed_python_runtime.md); `test_freshness_service_identity.py`. |
| K8 | Jeden request, JCS/base64url, limit 4 MiB, dokładny binding PREPARED do trzech wysłanych obiektów. Przed send odmowa → UNAVAILABLE; po rozpoczęciu send niejednoznaczność → OUTCOME_UNKNOWN, bez retry. | `freshness_runtime_ipc.py`: `_decode_response`, `_check_candidate_binding`, `verify_exact_candidate`; testy framing/binding/timeout/partial send. |

**Blokujące wymaganie to K4.** `capture_process_identity` może diagnostycznie
zwrócić `exe_sha256=None` po `PermissionError` przy `allow_missing_executable=True`
(`linux_service_ipc.py`, linie 67–101, 181). `_check_server` nadal odrzuca taki
wynik na linii 109, przed pierwszym send na linii 350. Opisuje to również
[obecny kontrakt klienta](stage9_linux_freshness_runtime_ipc.md), sekcja
„Przypięcie socketu i procesu”, oraz [propozycja dwóch UID](stage9_cross_uid_verifier_identity_proposal.md).
Grupa socketu, znany hash pliku na dysku, PIDFD i Rust nie przyznają prawa do
odczytu `/proc/PID/exe` między UID. Raport administratora nie jest bieżącym
odczytem klienta. Nie dopuszczać `None`, nie zastępować hasha wartością expected.

Trzeba rozróżnić K4 od mocniejszej własności badanej w #3252/#3253: wykluczenia
niezatwierdzonego kodu w pamięci. **Niezgodność hasha to odmowa spełnienia
przesłanki, nie dowód wykonania złośliwego kodu. Zgodność hasha nie dowodzi
integralności pamięci, bibliotek, parsera ani aktualnego posiadacza FD.**

Istnieją także niezależne blokady `IPC_BLOCKERS` w `linux_physical_ipc.py`:
`RUNTIME_IPC_APPLICATION_ADAPTER_UNAVAILABLE`,
`ACTUAL_RUNTIME_CLIENT_EXCHANGE_UNPROVEN`,
`RUNTIME_UNAUTHORIZED_IDENTITY_DENIAL_UNPROVEN`,
`DURABLE_CHALLENGE_REPLAY_PROTECTION_UNAVAILABLE`.
`require_ipc_qualification` nadal odmawia. AccountGenesis pozostaje poza zmianą;
jego stary transport/retry i drift przypiętych bajtów nie znikają od przyjęcia ADR.

## Formalny model zagrożeń

Zaufane są: jądro, systemowy manager działający jako root, administrator
instalacji i bootstrap, niezależnie zatwierdzony podpis wydania oraz kod startowy
wybranego verifiera. Zaufanie do kryptografii i retained credentials pozostaje
zgodne z obecnymi kontraktami. PostgreSQL i jego administrator są osobną granicą;
nie dowodzi się ich uczciwości przez uwierzytelnienie AF_UNIX.

Napastnik może wykonywać zwykły kod pod wskazanym UID, tworzyć pliki/sockety
w dostępnych mu miejscach i próbować dozwolonych syscalli. Nie ma capabilities,
uprawnień managera ani prawa do zapisu zatwierdzonych materiałów. Ochrona przed
obcym procesem UID verifiera jest wymaganiem, a nie założeniem, że taki proces
„nie może istnieć”. Nologin, dedykowane konto i brak crona zmniejszają drogi
wejścia, ale nie stanowią tego dowodu.

| Scenariusz | Wejście i wymagane uprawnienia | Możliwy skutek | Wymagana obrona / granica |
| --- | --- | --- | --- |
| T1: UID klienta | Przejęcie runtime/pluginu, zwykły proces pod `os_freshness_runtime`. | Dozwolone wejście do protokołu, fałszywe dane, DoS, próba podszycia pod verifiera. | K1/K3/K5 i tożsamość instancji serwera; ścisły parser i istniejące podpisy/selektory. Serwer uwierzytelnia UID klienta, nie jego kod. Przejęty runtime nadal może używać swojej roli DB. |
| T2: inny zwykły UID | Lokalne konto, próba connect, podmiany ścieżki lub sygnalizacji. | Podszycie, wyciek, przerwanie dostępności. | DAC/ACL przodków, grupa IPC, kernel credentials, chroniony systemd/cgroup, odmowa przed payload. Brak polegania wyłącznie na nazwie socketu. |
| T3: obcy proces UID verifiera | RCE innej usługi, błędne nadanie konta, obcy launcher; bez root. | Właściwy UID, ale zły proces; ptrace, sygnały, przejęcie FD; także własne połączenie PG jako rola verifiera. | Dokładny MainPID/cgroup/PIDFD, ochrona startu i ciągłości, brak cgroup delegation. Odrębna odmowa dostępu do PG; sama grupa/peer jej nie daje. DoS sygnałami pozostaje możliwy, jeśli nie obejmie go niezależna polityka. |
| T4: pliki, socket lub unit | Zapis do artefaktu/przodka, ACL, drop-in, bind mount, modyfikacja środowiska. | Uruchomienie innego kodu, błędne expected, podstawiony endpoint. | Root-owned niezapisywalne closure i przodkowie, niezależna weryfikacja podpisów przed startem, dokładny effective unit/argv/env/drop-ins, atomowa publikacja i K5. Możliwość takiego zapisu oznacza REJECT instalacji. Root może to zmienić i jest poza modelem. |
| T5: ingerencja przed startem | Same UID ptrace/TRACEME, process_vm, `/proc/PID/mem`, odziedziczone FD/mapowania przed seal. | Zmieniony proces może zainstalować poprawny filtr i zgłosić gotowość. | Ochrona od managera przez exec/loader: dla A/B kandydat to systemowa Yama `ptrace_scope=2` od bootu i brak CAP_SYS_PTRACE u napastników, zaufany rodzic i zamknięte dziedziczenie. Samo późne dumpable=0, TracerPid=0 lub AppArmor tracee nie wystarcza. READ_FSCREDS wymaga osobnej oceny. |
| T6: exec/fork i FD po starcie | Dozwolony syscall, odziedziczony listener/ring lub SCM_RIGHTS/pidfd_getfd. | PID/start/cgroup pozostają zgodne, ale wykonuje inny kod lub dane odbiera inny proces. | Seal przed bind/listen: rzeczywiste zakazy exec/execveat, fork/vfork/clone/clone3, zmian credentials/namespaces/dumpable, przekazywania FD i io_uring; brak wcześniejszych dzieci, ringów, dzielonej pamięci i obcych kopii FD. PIDFD nie dowodzi aktualnego wykonawcy ani filtra. |
| T7: podatność samego verifiera | Złośliwy dokument, błąd parsera/crypto/biblioteki, także błąd unsafe w Rust. | Kod lub błędna decyzja w uwierzytelnionej instancji; zgodny hash może pozostać. | Minimalny parser, bounds, audyt, fuzzing/wektory, izolacja i najmniejsze prawa. To ryzyko rezydualne zaufanego komponentu; A/B/C nie dowodzą ogólnej odporności na dowolne RCE. Seccomp nie sprawdza semantyki dozwolonych operacji ani Python bytecode. |
| T8: root lub jądro | Przejęcie administratora/kernel, równoważne capabilities lub managera. | Podmiana wszystkich dowodów, pamięci i polityk. | Jawnie poza granicą gwarancji. Raport/sygnatura lokalnego obserwatora nie atestuje uczciwości hosta. |

Dla T3 istnieje konkretna dodatkowa blokada: `deployment/postgresql/pg_ident.conf`
mapuje samą nazwę OS, `pg_hba.conf` dopuszcza peer, a
`PostgreSQLVerifierConnection.connect` sprawdza role i lokalność, nie cgroup ani
kod procesu. Obcy proces tego UID może zatem otrzymać rolę
`freshness_crypto_verifier`, jeżeli dosięgnie publicznego socketu PG.
Nie uznawać takiego dostępu za rozwiązany przez wybór A, B lub C.

Kandydat wymagający osobnego projektu to prywatny socket PG za root-owned
przodkiem, udostępniany wyłącznie uprawnionym przestrzeniom montowań, bez TCP
i alternatywnego publicznego endpointu. Musi objąć `/proc/PID/root`, cwd, FD,
setns i cały start. Yama2 blokuje ATTACH, ale nie każdy READ_FSCREDS; późne
`dumpable=0` nie dowodzi ochrony dostępu przez procfs przed seal. Execute-only
ELF z #3253 jest tylko kandydatem NOT_PROVEN. Ten ADR **nie zatwierdza** zmiany
ścieżki PG ani takiego mechanizmu. Brak zaakceptowanego rozwiązania i natywnej
odmowy T3 zatrzymuje kwalifikację autorytetu, nawet po dodatnim dowodzie tożsamości.

## Trzy warianty i dokładna podstawa zaufania

### A. Utwardzony Python i dedykowany UID

TCB obejmuje host/manager/bootstrap, podpisany interpreter i całą aplikację,
stdlib, wheels, rozszerzenia, OpenSSL/libpq oraz kontrolę ich ładowania.
Systemd zapewnia dokładne UID/GID, brak dodatkowych grup/capabilities,
NoNewPrivileges, chronione pliki i brak delegacji cgroup. Konto nie ma logowania,
user managera ani wspólnych zadań. Instalacja i zależności są niezapisywalne dla
obu usług, bez user-site, `.pth`, pluginów i importów z cwd/środowiska.
AppArmor musi być rzeczywiście enforce, z zamkniętym inwentarzem; seccomp musi
obejmować właściwy ABI i drogi FD/exec/io_uring. Same nazwy polityk nie są dowodem.

Przy niedostępnym `/proc/PID/exe` klient mógłby ufać wyłącznie **nowej przesłance**:
zatwierdzony manager uruchomił dokładny zweryfikowany Python i closure; kernel
wiąże połączenie z MainPID tej instancji; niezależna polityka i zaufany startup
zapewniają ciągłość. Snapshot, hash interpretera, SO_PEERSEC lub `sealed=true`
od serwera nie zastępują tej przesłanki. Alternatywą jest dodanie C.

Pozostają Python w pamięci, błędy rozszerzeń, pre-start, delegacja FD i T3/PG.
Przy wymogu wykluczenia dowolnego bytecode po kompromitacji interpretera: REJECT.
Przy świadomym zaufaniu całej aplikacji: możliwość wymaga nowego kontraktu i
kwalifikacji, NOT_PROVEN. Protokół/SQL mogą pozostać obecne, lecz bootstrap
używa subprocess OpenSSL, a produkcyjny serwer łączy PG przed listenerem;
nie wolno przyjąć modelowego filtra sondy jako profilu tej aplikacji.

### B. Minimalny natywny verifier — rekomendacja warunkowa

Jedna usługa bez interpretera, pluginów, powłoki i subprocess crypto. Obejmuje
krytyczny parser, weryfikację podpisów/retained credentials i istniejącą granicę
przygotowania PostgreSQL. Część Python poza tą granicą nie może autoryzować
wyniku ani mieć praw przygotowania. Rust jest kandydatem implementacyjnym;
unsafe/FFI, biblioteki kryptograficzne, linker/loader i dostawcy pozostają w TCB.
Wybrać jeden audytowalny build i zamknięty zestaw zależności, statyczny jeśli
wykonalny; nie deklarować closure bez inwentarza bibliotek i zasobów.

**Nowy, proponowany predykat klienta**, wspólny dla A/B, jest koniunkcją,
nie zestawem fallbacków. W A program i closure obejmują Python i aplikację,
w B natywny obraz i jego zależności; technologia nie zmienia wymagań tożsamości:

1. Niezależnie zaakceptowany manifest wydania wiąże dokładny program/closure,
   skuteczny unit, argv/env, rewizję polityk i ich wymagania platformowe.
   Zaufany bootstrap weryfikuje je przed wykonaniem kodu nowej usługi.
2. Klient odczytuje systemowy manager przez chroniony system bus, z walidacją
   adresu/owner i polityk D-Bus/Polkit; bez user bus i caller-selected endpointu.
   Wymaga dokładnego unitu bez drop-ins/delegacji, aktywnej gotowej instancji,
   MainPID=ExecMainPID, cgroup, UID/GID i zatwierdzonej epoki uruchomienia.
   Istniejący `freshness_service_identity` nie implementuje tej nowej ścieżki.
3. K1/K3/K5 pozostają; PID z tego samego połączonego AF_UNIX musi zgadzać się
   z managerem. Tożsamość obejmuje boot ID, InvocationID, MainPID, start ticks
   i zatwierdzoną instalację; nowy boot/restart unieważnia poprzednie expected.
   Brak dostępu do któregokolwiek wymaganego faktu kończy się odmową.
4. Start jest chroniony przed T5. Dla obu wariantów proponuje się Yama2 od bootu
   dokładnej VM, brak CAP_SYS_PTRACE u zwykłych procesów i zaufanego rodzica.
   Wpływ globalnej polityki, jądro i jej niezmienność wymagają decyzji właściciela.
   Nie wnioskuje się tego z profilu AppArmor ofiary ani z końcowego dumpable.
5. Przejrzany kod startowy przed gotowością i bind/listen usuwa niezatwierdzone
   FD/dzieci/mapowania, ustawia i sprawdza dumpable=0 oraz NNP, instaluje
   domyślnie odmawiający filtr dla zatwierdzonego ABI. Nie ma socket activation.
   Po seal zakazuje T6, także sendmsg/sendmmsg/recvmsg/recvmmsg i trzech syscalli
   io_uring; nie zostawia alternatyw przez obce ABI/x32. Dokładna allowlista
   musi działać z rzeczywistym crypto i PG, bez dopisywania szerokich wyjątków.
   Gotowość pochodzi z tej ścieżki kodu; komunikat serwera sam jej nie poświadcza.
6. Klient ponawia kontrolę instancji/epoki i przypiętego kanału przed payload
   i po odpowiedzi. Zakazy T5/T6 zamykają przerwę między obserwacjami; same
   dwa zgodne snapshoty nie zamykają wyścigu. K8 pozostaje bez zmian.

Klient **ufa wykonaniu zatwierdzonego kodu startowego i egzekwowaniu polityk**,
a nie niezależnej inspekcji wszystkich instrukcji/pamięci lub dokładnego filtra
w każdej wymianie. Tryb seccomp=2 z procfs sam nie dowodzi treści filtra.
To jest jawna zmiana modelu względem K4. Poprawna implementacja tego predykatu
pozostaje NOT_PROVEN; obecny klient nadal odmówi przy `exe_sha256=None`.

Wymagana zgodność funkcjonalna: te same trzy obiekty, `!I`+JCS, base64url,
4 MiB, Ed25519, domeny i preimages, selektory retained credentials, PREPARED
oraz binding kandydata. Bez zmiany algorytmów, domen, schematu czy obecnych pinów.
Nie używać własnej kryptografii. Wektory różnicowe muszą objąć duplikaty JSON,
Unicode, bool/int, zakresy liczb, base64url i błędne podpisy. Zachować role,
transakcje, trwałość przygotowania i klasyfikację niejednoznacznego COMMIT;
brak reconnect/retry po rozpoczęciu send. Przeniesienie kodu nie zapewnia replay.

Wydajności nie zakłada się z języka: zmierzyć p50/p95/p99, przepustowość,
pamięć, startup i czas zapytań PG na tym samym korpusie. Bramką jest istniejący
deadline bez zwiększania limitu, brak nieskończonej alokacji i brak rozbieżności
wyników. Twardy dodatkowy budżet regresji musi zostać zapisany w etapie 1,
zanim pomiar stanie się podstawą ACCEPT. T3/PG pozostaje osobnym warunkiem.

### C. Niezależny obserwator — fallback

Mały natywny komponent, osobny od Pythona, kontrolowany przez root, o stałym
celu i ograniczonym API. Jeżeli potrzebuje CAP_SYS_PTRACE, dostaje ją tylko on,
z ograniczeniem zakresu przez własną politykę, sandbox i walidację celu;
sama capability nie ogranicza się automatycznie do jednego PID. Python nie
otrzymuje capability, setuid ani ogólnego API odczytu `/proc`.

Obserwator otrzymuje dokładnie połączony socket przez osobny uwierzytelniony
kanał SCM_RIGHTS i sam pobiera jego SO_PEERCRED/SO_PEERPIDFD. Nie ufa PID,
ścieżce ani pidfd podanym jako treść żądania. Sprawdza managera/epokę,
chroniony inwentarz i bieżący obraz; odpowiedź wiąże nonce żądania, boot,
instancję, endpoint i wynik. Kanał klient–obserwator ma własną ścisłą tożsamość,
limity i odmowy, bez podpisu/self-report uznawanego za dowód uczciwości hosta.

Jednorazowe poświadczenie przed send jest za słabe: exec i delegacja FD mogą
nastąpić później. C musi korzystać z wykazanej ochrony startu/ciągłości jak A/B
albo mediować dokładną wymianę z udowodnioną ochroną wszystkich przejść.
Wstrzymanie procesu na czas hashowania nie rozwiązuje automatycznie dalszego
odbioru. Utrzymywanie ptrace przez całą wymianę zwiększa TCB i wymaga modelu
exec/fork/threads/signals; nie jest gotowym rozwiązaniem tego ADR.
Jeżeli helper sam łączy się z verifierem i przekazuje payload, serwer widzi UID
helpera zamiast runtime. To wymaga osobnej zmiany guardu klienta po stronie
serwera; nie wolno ukryć jej pod nazwą „obserwacja”.
Klientowi nie wolno wysłać payload po wygaśnięciu/przerwaniu sesji dowodowej.
Przekazywanie klientowego FD do helpera musi być oddzielone od zakazu delegacji
FD przez verifiera. Retained/replayed verdict, zły FD, zgon obserwatora i race
kończą się fail closed. Hash Python nadal nie mierzy całej aplikacji/pamięci.

## Ocena ACCEPT / REJECT / NOT_PROVEN i koszt

ACCEPT oznacza dowód **wskazanej własności**, REJECT jej niespełnienie w podanym
modelu, NOT_PROVEN brak wymaganych dowodów. Rekomendacja nie jest ACCEPT
implementacji. W tej tabeli nie ma dodatniej kwalifikacji produkcyjnej.

| Własność | A | B | C |
| --- | --- | --- | --- |
| Zgodność z obecnym K4 przy braku odczytu `/proc/exe`, bez zmiany kontraktu | REJECT | REJECT | REJECT: obecny guard nie przyjmuje delegowanego dowodu |
| Nowa podstawa tożsamości po jawnej decyzji | NOT_PROVEN: start+closure+instancja | NOT_PROVEN: start+zamknięty ELF+instancja | NOT_PROVEN: uwierzytelniony obserwator+ten sam kanał+ciągłość |
| Pełna hybryda #3253 wobec arbitrary Python w pamięci | REJECT | Nie dotyczy interpretera; brak ogólnego dowodu RCE | REJECT jako gwarancja uzyskana tylko przez pomiar hasha Python |
| Ochrona przed obcym procesem tego samego UID, od startu i po seal | NOT_PROVEN; Yama2 i seal wymagają dowodów | NOT_PROVEN; Yama2 i seal wymagają dowodów | NOT_PROVEN; helper sam nie usuwa T5/T6 |
| Pełna kwalifikacja AF_UNIX/systemd/PG na dwóch UID | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN |
| Istniejąca odmowa klienta przy brakującym executable | ACCEPT: K4/test negatywny | Ten sam guard nadal obowiązuje | Ten sam guard nadal obowiązuje |

Liczba granic oznacza miejsca wymagające niezależnej umowy/testu, nie estymatę
terminu. Komponenty obejmują nowe lub modyfikowane elementy przyszłej realizacji;
nie są zakresem tego dokumentacyjnego PR.

| Wariant | Zmiany kontraktu produkcyjnego po zgodzie | Utrzymanie, regresje i niezależne testowanie | Złożoność |
| --- | --- | --- | --- |
| A | Ta sama nowa przesłanka K4/K6/K7; dokładny start/seal i closure; ochrona T3/PG oraz T5/T6. | Zachowuje istniejący parser/crypto/SQL, ale wymaga audytu importów/FFI/ABI i rekwalifikacji zależności. Rozmiar closure i koszt utrzymania wymagają pomiaru; testy tożsamości takie same jak B. | 5 granic: wydanie→start, start→seal, manager→klient, proces→kanał, proces→PG; ok. 6 komponentów: closure/builder, bootstrap, unit/polityki, klient, harness, izolacja PG. |
| B | Ta sama nowa przesłanka K4/K6/K7 i ochrona T3/PG, T5/T6; osobne zatwierdzenie natywnego wdrożenia. Bez samoczynnej wymiany pinów. | Mniejsze closure jest hipotezą. Dochodzi migracja parsera/crypto/PREPARED/PG; konieczne wektory różnicowe, fuzzing, audyt unsafe/FFI i dostawców oraz te same kernelowe testy instancji. | Te same 5 granic; ok. 7 komponentów: natywna usługa, builder/inwentarz, bootstrap, unit/polityki, klient, harness, izolacja PG. |
| C | Uwierzytelniony kanał obserwatora, delegacja bieżącego pomiaru zamiast odczytu klienta, binding sesji i ciągłość; polityki helpera i PG. | Największy przyrost uprzywilejowanego TCB; regresje FD/races/lifetime i aktualizacji jądra. Niezależny klient adversarialny, zły FD/nonce/epoch, awaria helpera i testy jego uprawnień. Python closure nadal pozostaje, jeśli pozostaje Python. | 7 granic: pięć bazowych oraz klient→helper i helper→proces; ok. 9 komponentów: elementy A plus helper, jego unit/polityka oraz osobny protokół/harness. |

### Wspólne kryteria wyboru A/B

Porównujemy ten sam model T1–T8: identyczną zaakceptowaną alternatywę K4,
T3 także wobec obcego procesu UID verifiera i PG, ochronę T5 od startu oraz
zakazy T6. Ten sam host, uprawnienia, korpus i progi; bez wyłączenia RCE tylko
dla B. Historyczny REJECT Python w pamięci nie rozstrzyga tego porównania:
przy absolutnym wymogu odporności na dowolne RCE żaden wariant nie ma dowodu.

| Hipoteza korzyści B | Wymagany porównywalny dowód A/B | Granica wniosku |
| --- | --- | --- |
| Mniej kodu i dróg ładowania w TCB | Niezależnie sprawdzony inwentarz zależności i mapowań obu usług; w B zero interpretera/pluginów, wykaz usuniętych dróg importu i nowych bibliotek/dostawców. | Sam mniejszy rozmiar, LOC lub liczba plików nie dowodzą mniejszego ryzyka; trzeba wskazać usuniętą drogę ataku. |
| Łatwiejsza ścisła izolacja | Ta sama macierz P1–P6 i odmowy T3/PG, dokładne allowlisty i audyt startu; wykazana realizacja uprzednio wybranego ograniczenia w B, którego A nie spełnia, bez nowych uprawnień lub słabszych odmów. | Sonda Rust dowodzi własnej ochrony; nie dowodzi niewykonalności ochrony A ani zgodności całej usługi. |
| Mniej ryzyka błędów pamięci i utrzymania | Przegląd osiągalnego kodu natywnego, unsafe/FFI i zależności obu wariantów oraz odtwarzalna próba aktualizacji/rekwalifikacji według wspólnego budżetu. | Python już ogranicza błędy pamięci kodu wysokiego poziomu. Rust nie usuwa błędów logiki, unsafe, bibliotek ani dostawców; liczba crate nie jest miarą bezpieczeństwa. |

Materiały tego ADR nie zawierają tych porównań: przewaga B jest **NOT_PROVEN**. Dodatkowy
koszt B obejmuje dobór/walidację implementacji Ed25519, JCS/base64url i typów,
preimages oraz bindingu PREPARED, a także sterownika PG, granic transakcji,
trwałości i niejednoznacznego COMMIT. Wymagane są niezależne wektory protokołu
(Python nie jest jedynym oracle), testy różnicowe i awarie PG; zero rozbieżności
przed zastąpieniem usługi. Przejście testów tożsamości nie dowodzi tej zgodności.

**Wybieramy A**, gdy spełnia wspólne warunki i budżety, a B nie ma wykazanej
korzyści uzasadniającej koszt migracji. **Wybieramy B** dopiero, gdy spełnia te
same warunki i niezależny przegląd potwierdzi co najmniej jedną wcześniej
zdefiniowaną korzyść z tabeli oraz akceptowalne ryzyko migracji. Sam język nie
jest korzyścią dowodową. **Oba: NOT_PROVEN**, gdy brakuje dowodów; niespełnienie
obowiązkowego warunku daje REJECT danego wariantu. Nierozwiązane K4/T3/T5/T6
nie uzasadniają migracji do B.

**Przed rozpoczęciem portu produkcyjnego** wymagane są: zatwierdzenie wspólnej
granicy i progów, natywny IDENTITY_ACCEPT, porównawcze artefakty uzasadniające B
z późniejszej, odrębnej próby TEST_ONLY, niezależny przegląd ryzyka migracji
oraz zatwierdzony korpus i plan zgodności parsera/crypto/PREPARED/transakcji PG.
Przed wdrożeniem dodatkowo obowiązują pełna równoważność i AUTHORITY_ACCEPT;
żadna z tych zgód nie zmienia automatycznie pinów ani pozostałych bramek.

**Minimalna konieczna zmiana** wspólna dla A/B: właściciel jawnie zatwierdza
wersjonowaną alternatywę dowodową K4 „zatwierdzony start + dokładna instancja
managera + związany kanał + egzekwowana ciągłość” zamiast obowiązkowego live
hasha odczytanego przez klienta. K1/K3/K5/K8 pozostają. K6/K7 dostają dokładną
wersję dla tej alternatywy. Dla C minimalna alternatywa to live pomiar wykonany
przez zaufanego helpera dla tego samego kanału i okresu, nadal z ciągłością.
Nie są to zgody na `exe=None`, pominięcie guardu lub automatyczny fallback.

Jeżeli właściciel wymaga równocześnie starego K4 bez delegacji, różnych UID,
braku uprawnień odczytu klienta oraz braku nowej przesłanki, rozwiązanie pozostaje
zablokowane. Jeżeli wymaga wykluczenia dowolnego RCE w zaufanym verifierze,
żaden wariant nie ma dowodu. Osobno trzeba rozstrzygnąć pełny zakres T3/PG.

## Plan: najwyżej trzy etapy z zamkniętymi bramkami

### 1. Zatwierdzenie granicy i kryteriów — bez nowego eksperymentu

Artefakt: zaakceptowana przez właściciela i niezależnego recenzenta wersja ADR,
lista K1–K8/T1–T8 ze wskazanym obowiązującym kontraktem, niezmienny korpus
wektorów/testów, budżet wydajności oraz osobna decyzja o pojedynczej VM.
Korpus musi mieć jawnie zdefiniowane oczekiwane wyniki, zero dopuszczonych
rozbieżności semantycznych i brak pominięć obowiązkowych przypadków.

Wymagane decyzje: mechanizm tożsamości (wspólna przesłanka A/B albo C), osobno
kryteria wyboru technologii A/B; delegacja/zastąpienie K4; zaakceptowanie ryzyka
podatności wewnątrz TCB bez roszczenia pełnej atestacji pamięci; polityka ochrony
startu (dla A/B kandydat Yama2 i jej wpływ na host); dokładny zakres T3/PG i
mechanizm, który ma być kwalifikowany; lifecycle/rewokacja po restarcie;
limity wydajności. Jeżeli PG pozostaje publiczne i peer-only, zapisać
**REJECT pełnej kwalifikacji autorytetu**, bez wyłączenia T3 z raportu.
Wybór technologii może pozostać NOT_PROVEN do porównania; akceptacja mechanizmu
tożsamości nie upoważnia do przepisywania produkcyjnej usługi.

**ACCEPT:** istnieją wszystkie powyższe jawne decyzje i podpisane/zaakceptowane
kryteria, z rozdzieleniem IDENTITY_ACCEPT od pełnej kwalifikacji autorytetu.
Można zatwierdzić wyłącznie tożsamość, zachowując blokadę PG i produkcji.
**REJECT/STOP:** brak decyzji o K4 lub T7, sprzeczne wymagania, próba zastąpienia
brakującego dowodu założeniem, albo brak skończonego oracle testów.
Bez ACCEPT tego etapu nie powstaje następna sonda.

### 2. Izolowany prototyp TEST_ONLY — pierwszy prawidłowy dowód

Pierwszy krok po ACCEPT etapu 1: minimalny endpoint z syntetycznym challenge
i **osobny klient TEST_ONLY**, sprawdzające wyłącznie nową podstawę tożsamości.
Endpoint może być natywną sondą Rust; nie przesądza to wyboru B dla aplikacji.
Zachować produkcyjne `_check_server` i negatywną regresję `exe=None`.
Prototyp ma własne syntetyczne wydanie/stałe, bez parsera i kryptografii dokumentów,
PREPARED, SQL, AccountGenesis i zmian zamrożonych materiałów. P7 jest późniejszą
próbą zgodności aplikacji, poza pierwszym dowodem tożsamości. Port parsera,
kryptografii i PostgreSQL nie jest warunkiem rozpoczęcia tego prototypu.

Artefakty: źródła i reproducible build/inwentarz, dokładny predykat i maszyna
stanów start→seal→ready→jedna wymiana→zamknięcie, raport każdego przypadku
z oczekiwaniem/obserwacją, osobne liczniki bajtów wysłanych przez poprawnego
klienta i odczytanych/przetworzonych przez serwer oraz licznik przygotowań,
instrukcja reprodukcji rootless i pakiet do etapu 3. Napastnik może wysłać dane
do kolejki socketu; nie deklaruje się braku takich bajtów. Stan DONE prototypu
nie jest NATIVE PASS.

| ID | Obowiązkowa próba | Mierzalny oracle |
| --- | --- | --- |
| P1 | Właściwy endpoint, instalacja, manager/epoka i związany kanał. | Dokładnie jedna poprawna odpowiedź dla jednego żądania, tożsamość z nowego predykatu; bez przypisania expected do brakującej obserwacji. |
| P2 | Zły UID klienta i zły UID serwera; obcy proces UID verifiera poza właściwą jednostką, także ten sam ELF. | Zły serwer: poprawny klient wysyła 0 bajtów aplikacyjnych. Zły klient: serwer odczytuje/przetwarza 0 bajtów aplikacyjnych i wykonuje 0 przygotowań. Przyczynę odmowy potwierdza kontrola pozytywna. Sam connect nie jest payload. |
| P3 | Zły unit/drop-in/argv/env/inwentarz, podmiana socketu/przodka, zły MainPID/cgroup, stara epoka po restarcie. | Brak uwierzytelnienia i 0 payload przed send; restart nie akceptuje starego expected. |
| P4 | Exec/execveat, fork/vfork/clone/clone3, SCM_RIGHTS w obie strony, pidfd_getfd, io_uring i obce ABI/x32. | Rzeczywista odmowa przypisana mechanizmowi; brak obcego odbiorcy/nowego wykonawcy. Stare zgodne credentials nie wystarczają. Brak syscalla = BLOCKED, nie PASS. |
| P5 | Ptrace/process_vm/proc-mem przed pierwszą instrukcją i podczas loadera; wcześniejsze FD/mapowania/dzieci. | Kontrola dowodzi osiągalności próby, właściwy wariant odmawia lub odrzuca start przed ready. Końcowy TracerPid=0 nie jest oracle. |
| P6 | Śmierć/restart procesu, awaria dowodu/helpera, timeout, partial send, złe powiązanie kanału/epoki; binding PREPARED później w P7. | Przed send UNAVAILABLE/0 payload; po rozpoczęciu send OUTCOME_UNKNOWN, bez kolejnej próby i bez twierdzenia o rollbacku. |
| P7, późniejsza kwalifikacja aplikacji | Parser/crypto i PREPARED na korpusie zgodnym i złośliwym. | Zero różnic względem uzgodnionego oracle, dokładne limity i podpisy, żadnego kodu/deserializacji programu z danych. |
| P8, obowiązkowe dla C | Fałszywy helper, inny przekazany socket FD, stary/powtórzony werdykt, zły nonce także w tej samej epoce, przerwanie sesji dowodowej przed send. | Poprawny klient odmawia przed payload: 0 wysłanych bajtów aplikacyjnych; żaden werdykt dla innego kanału/żądania nie autoryzuje wymiany. |

Modele rootless pomagają sprawdzić parser i maszynę stanów. Nie wolno z ich
mocków uzyskać dodatniego dowodu dwóch UID/systemd. P1/P2 na rzeczywiście dwóch
UID, P4/P5 zależne od hosta pozostają NOT_RUN/BLOCKED do zatwierdzonego etapu 3.
Etap 2 może zakończyć się **ACCEPT_TEST_ONLY** tylko dla jawnie wykonanego zakresu;
nie ogłasza pierwszego natywnego dowodu, jeżeli nie było różnych UID.

**REJECT/STOP:** poprawny klient wysłał payload mimo odmowy serwera lub serwer
odczytał/przetworzył payload klienta o odrzuconej tożsamości; konieczny jest bypass/monkeypatch
obserwacji tożsamości, nowe szerokie uprawnienie Pythona, niewyjaśniona rozbieżność
korpusu albo rozszerzenie TCB niezatwierdzone w etapie 1. Nie dopisywać kolejnego
modelu jako naprawy bez ponownej decyzji.

### 3. Natywna kwalifikacja na osobno zatwierdzonej VM

Dopiero po osobnej zgodzie na konkretną jednorazową VM: dokładne Ubuntu 24.04,
rewizja jądra/ABI, systemd 255 i AppArmor z zapisanym stanem enforce. Używać
rzeczywistych odrębnych UID, właściwego systemowego managera, AF_UNIX/PIDFD oraz
tymczasowego PostgreSQL z kontami i danymi wyłącznie TEST_ONLY. Bez kontaktu
z produkcyjnymi poświadczeniami i bazą. Nie instalować polityk na bieżącym hoście.
Dla pierwszego IDENTITY_ACCEPT nie podłączać PG ani komponentów aplikacji;
P7 i część PostgreSQL pozostają NOT_RUN do późniejszej kwalifikacji autorytetu.

Artefakty: zgoda i identyfikator VM, dokładny commit/build/manifest, inwentarz
OS/kernel/systemd/LSM/sysctl, rzeczywiste UID/GID/grupy, effective unit oraz
boot/InvocationID/MainPID, chronione endpointy i obserwacje kanału, raport P1–P6
oraz P8 dla C bez mocków i cleanup; dla AUTHORITY_ACCEPT także P7 i log PG. Dowody muszą zawierać
pole oczekiwanego wyniku, rzeczywistą przyczynę odmowy i niezależną kontrolę
osiągalności; zastane EPERM/EINVAL nie jest automatycznie zasługą filtra.

**IDENTITY_ACCEPT:** P1 i wszystkie obowiązkowe negatywy tożsamości P2–P6
oraz P8 dla C wykonane bez pominięć, właściwe liczniki P2/P8 równe 0, poprawna epoka,
nieprzerwana ochrona startu/FD i transportowa część K8: jedna wymiana, deadline,
klasyfikacja niejednoznaczności i brak retry. Pełne K8/P7 nie jest warunkiem
tego dowodu; produkcyjne K8 pozostaje bez zmian. Jest to pierwszy prawidłowy
natywny dowód tożsamości wybranego kontraktu, bez obchodzenia starego guardu.

**AUTHORITY_ACCEPT dla zakresu VM:** dodatkowo pełne K8/P7 bez rozbieżności, rzeczywisty
PG peer/role, poprawne przygotowanie właściwego kandydata; runtime nie wywołuje
prepare, verifier nie uzyskuje CAS, inni zwykli UID **i obcy proces UID verifiera**
nie uzyskują uprzywilejowanego dostępu do PG. Zweryfikować niezmienność stanu
po błędnym podpisie/selektorze, timeout/niejednoznaczny COMMIT oraz trwałość po
kontrolowanym restarcie testowego klastra. Wyniki i zasoby mieszczą się
w uprzednio przyjętym budżecie, bez zwiększenia deadline lub progów CI.
Jeżeli próba awarii nie rozstrzyga COMMIT, raportuje nieznany wynik i nie retry.

**REJECT/STOP:** jakiekolwiek fałszywe uwierzytelnienie, przyjęty stary okres,
przekazany kanał do obcego odbiorcy, niechroniony start, pominięty obowiązkowy
test właściwego zakresu (P7 obowiązkowe dopiero dla autorytetu) albo dostęp T3
do roli PG. Brak mechanizmu hosta oznacza BLOCKED/NOT_PROVEN,
a nie zgodę na słabszy test. Zatrzymać próbę, zachować dowody i zakończyć według
zatwierdzonego cleanup; nie rozszerzać polityk eksperymentalnie.

AUTHORITY_ACCEPT na VM nie zamyka pozostałych IPC_BLOCKERS, lifecycle/replay,
pinów ani integracji AccountGenesis i nie oznacza PRODUCTION_GO.
Produkcję może odblokować wyłącznie odrębna pełna procedura kwalifikacji.

## Walidacja ADR i granice bieżącej zmiany

Ten PR dodaje wyłącznie ten ADR. Nie zmienia klienta/verifiera, AccountGenesis,
podpisów/digestów/pinów, PostgreSQL/poświadczeń, CI/progów ani
`deployment/stage9_current_status.json`. Nie uruchamia usług, sond
administracyjnych, nowych workflow ani polityk hosta. Draft PR, bez merge.

Odtwarzalne kontrole dokumentacyjnej zmiany (wynik dla końcowego commita należy
do opisu PR; samo wymienienie polecenia nie oznacza PASS):

```bash
git diff --check
python -m pre_commit run ruff-format --all-files
python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
python scripts/quality/betterleaks_hook.py docs/architecture/stage9_freshness_verifier_identity_adr.md
python -m pytest -q tests/security/test_freshness_runtime_ipc.py tests/security/test_freshness_runtime_ipc_identity.py tests/security/test_freshness_service_identity.py tests/security/test_linux_service_ipc.py tests/deployment/test_linux_physical_ipc.py tests/architecture/test_m05_account_genesis_freshness_authority_production_local_authentication_gate.py
```

Przegląd kontraktów i powyższe regresje potwierdzają zachowanie obecnej odmowy,
nie implementują nowej przesłanki. Kontrola odnośników względnych i diffu ma
potwierdzić wyłącznie pojedynczy plik dokumentacji i brak niepowiązanych zmian.
CI trzeba odczytać dla HEAD nowego Draft PR; queued/pending nie jest PASS ani
powodem restartowania workflow. Brak dowodu z VM pozostaje NOT_RUN.
