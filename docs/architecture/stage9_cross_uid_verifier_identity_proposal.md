# Stage 9 — tożsamość verifiera między odrębnymi UID

Status: **DECYZJA ARCHITEKTONICZNA DO ZATWIERDZENIA; BLOCKED**.
Analiza dotyczy `main` `a60be5946a02cfad317ec895b2f04ef7f63407e2`,
po scaleniu PR #3250, 5 listopada 2026 r. Nie zmienia produkcyjnego
uwierzytelniania. `STAGE9_PRODUCTION_READY=NO`, `NO MERGE`.

## Rozstrzygnięcie i model zagrożeń

Zwykły `os_freshness_runtime` przy rzeczywiście innym UID verifiera nie może
spełnić obecnego wymogu odczytu jego `/proc/PID/exe` w standardowym modelu
ptrace. Wspólna grupa socketu tego nie zmienia. Kernelowe poświadczenia,
pidfd, metadane systemd i podpis plików nie uzupełniają brakującego dowodu
aktualnego obrazu ani rzeczywistego odbiorcy deskryptora. Obecna odmowa
przed wysłaniem żądania jest prawidłowym wynikiem.

Rozpatrujemy lokalnego napastnika jako runtime lub innego użytkownika,
podstawienie usługi i handlera oraz uruchomienie innego programu pod UID
verifiera. Root instalacji, jądro, systemd i niezależni zatwierdzający wydanie
są zaufani. Nie zakładamy, że napastnik kontrolujący root pozostawia uczciwe
procfs, D-Bus, politykę jądra lub podpisany kontrakt. Osobno trzeba określić,
czy wymagane jest rozpoznanie zatwierdzonego procesu, czy także odporność
na wykonanie dowolnego kodu wskutek przejęcia jego pamięci. Żaden z opisanych
odczytów nie jest atestacją całej pamięci działającego Pythona.

Rekomendacja: zachować wszystkie obecne odmowy i przed integracją zatwierdzić
zmianę podstawy zaufania na **zatwierdzony start oraz egzekwowaną ciągłość**
(wariant A poniżej). Najpierw wykonać małą próbę odmowy z istniejącym klientem.
Wariant A jest projektem do niezależnej akceptacji, a jego skuteczność wymaga
osobnego prototypu i negatywnych prób jądrowych. Nie jest wdrożonym rozwiązaniem.
Jeżeli wymagany pozostaje niezależny bieżący pomiar wykonywanego kodu,
bez zmiany uprawnień lub granicy zaufania zadanie pozostaje niewykonalne.

## Co rzeczywiście widzi nieuprzywilejowany klient

| Obserwacja | Dostępność między UID | Dowód i ograniczenie |
| --- | --- | --- |
| `SO_PEERCRED` | Na połączonym AF_UNIX bez `CAP_SYS_PTRACE`; osobno obowiązuje DAC ścieżki | Zapisane przez jądro PID/UID/GID. Dla klienta stream poświadczenia pochodzą z `listen()`, a nie z każdego `accept()` lub `recv()`. Nie mierzy kodu ani grup dodatkowych. |
| Socket-bound `SO_PEERPIDFD` | Linux od 6.5; trzeba sprawdzić rzeczywisty kernel/ABI | Uchwyt do tego samego zapisanego `sk_peer_pid`; eliminuje pomylenie z później ponownie użytym numerem PID. Nie zmienia uprawnień procfs, nie wykrywa exec i nie identyfikuje późniejszego posiadacza FD. Brak obsługi oznacza odmowę, bez fallbacku do `pidfd_open(PID)`. |
| `/proc/PID/exe` | Podlega `PTRACE_MODE_READ_FSCREDS`; przy rozbieżnych UID standardowo EACCES/EPERM | Otwarcie wskazuje plik obrazu wykonywalnego w chwili obserwacji. Hash Pythona nie identyfikuje skryptu, importów ani całej pamięci. Dwa odczyty nie wykluczają zmiany i powrotu pomiędzy nimi. |
| `/proc/PID/stat`, pole 22 | Zwykle czytelne także przy odmowie `exe`; możliwe ograniczenia hidepid/LSM | Czas utworzenia procesu w tickach od bootu, pozostający taki sam po exec. Nie wszystkie pola stat mają ten sam warunek ptrace; pól oznaczonych `[PT]` nie należy utożsamiać z polem 22. |
| `status`, `cgroup`, `cmdline` | Zwykle odczytowe; zależą od mountów procfs, namespace i LSM | Credentials i grupa kontrolna są przydatne do korelacji. Cgroup przechodzi na potomka i pozostaje po exec. Cmdline/argv są mutowalne. Żadne z tych pól nie identyfikuje wykonywanej aplikacji. |
| Systemowy D-Bus systemd | Domyślna polityka systemd 255 dopuszcza `Properties.Get/GetAll`; sprawdzić w VM jako runtime | MainPID/ExecMainPID, ControlGroup i aktywny okres usługi korelują połączenie ze startem. ExecStart opisuje zleconą komendę i jej ostatnie wykonanie; nie mierzy każdego późniejszego exec. |
| Podpisany inwentarz runtime/aplikacji | Pliki mogą być dostępne tylko do odczytu bez nowych capabilities | Potwierdza zatwierdzone bajty i ich pochodzenie w zaufanym instalatorze. Nie dowodzi załadowania właśnie tych bajtów ani ciągłości ich wykonywania. |

W `__ptrace_may_access()` zgodność obejmuje UID/GID real/effective/saved celu
względem poświadczeń sprawdzającego lub odpowiednią `CAP_SYS_PTRACE` w user
namespace celu; dalej obowiązują dumpable i LSM. Samo `ptrace_scope=0`,
`PR_SET_PTRACER`, dumpable=1 lub grupa IPC nie usuwa podstawowej rozbieżności
UID. Podniesienie dumpable nie jest rozwiązaniem problemu.

D-Bus odpytywać przez stały lokalny system bus, bez dziedziczonego
`DBUS_SYSTEM_BUS_ADDRESS`, z zaufanymi narzędziami i chronioną ścieżką busa.
Nie ufać metadanym systemd dostarczonym przez verifiera. Brak pola lub prawa
odczytu oznacza odmowę. `ExecMainStartTimestampMonotonic` ma jednostkę czasu
systemd, a stat22 ticki od bootu; nie porównywać ich jako równych liczb.
Wymagane są boot ID, spójny okres usługi i uchwyt pidfd podczas korelacji.

## Scenariusze rozdzielające proces od handlera

| Zdarzenie | Co może pozostać zgodne | Czego obecne obserwacje nie dowodzą |
| --- | --- | --- |
| `execve` innego programu | PID, pidfd, start-time, cgroup, zapisane peer credentials i historyczny ExecStart | Bieżący odczyt exe może wykryć inny obraz, jeśli jest dostępny. Nie gwarantuje ciągłości pomiędzy kontrolami. |
| `execveat`, także `AT_EMPTY_PATH` | Jak dla execve; wykonanie może pochodzić z FD | Zakaz wyłącznie execve pozostawia drugą drogę. Ścieżka w argv nie rozstrzyga pochodzenia obrazu. |
| Exec tego samego interpretera z innym kodem/argv lub powrót do obrazu | Nawet hash exe i wszystkie statyczne pliki | Hash Pythona, argv ani odczyt przed/po nie stanowią dowodu nieprzerwanej aplikacji. |
| `fork` z odziedziczonym listenerem | Dla klienta PID i pidfd procesu, który wywołał listen; potomny handler ma zwykle ten sam UID/cgroup | Żądanie może czytać dziecko. Żywy rodzic i zgodny MainPID nie dowodzą jego obsługi. |
| Przekazanie listenera lub połączonego socketu przez `SCM_RIGHTS` | Tożsamość zapisana w sockecie i życie pierwotnego procesu | Przekazanie FD nie aktualizuje `sk_peer_pid`. Odczyt peer nie ustala, kto teraz wykonuje recv/send. |
| Start przez systemd socket activation | Peer może wskazywać twórcę/listenera socketu | Nie jest równoważny obecnemu samodzielnemu listen verifiera. Nowy wariant wymaga osobnego kontraktu. |

Otwarcie pidfd przez numer z MainPID jest użyteczne w kwalifikatorze usługi,
lecz nie zastępuje uchwytu pozyskanego z faktycznego socketu. Nawet przy
idealnej ochronie przed ponownym użyciem PID problem exec i delegacji pozostaje.
Kontrola po wysłaniu nie cofa ujawnionego żądania ani przygotowania w bazie;
niejednoznaczny późny wynik pozostaje `OUTCOME_UNKNOWN`, bez retry.

## Konkretne warianty architektoniczne

| Wariant | Granica i model zagrożeń | Uprawnienia / koszty | Weryfikacja i decyzja |
| --- | --- | --- | --- |
| **A. Zatwierdzony start i nieodwracalnie zamknięty proces przed listen** | Zaufany, podpisany launcher/aplikacja ustanawia ograniczenia przed udostępnieniem endpointu. Klient ufa tej kolejności i systemd, następnie koreluje socket-bound PIDFD, UID/GID, okres usługi i dostępne procfs. Chroni przed zastąpieniem obrazu i delegacją, jeśli wszystkie drogi są egzekwowane. Nie atestuje pamięci po RCE. | Runtime i verifier bez nowych capabilities oraz bez nowego uprzywilejowanego procesu. Istniejący root zatwierdza instalację/kontrakt. Wymaga zmiany bootstrapu/aplikacji i osobno zatwierdzonych nowych bajtów. | Preferowany kierunek do zatwierdzenia. Dokładny filtr i zamknięcie aplikacji wymagają prototypu; obecna jednostka tego nie zapewnia. Brak któregoś warunku utrzymuje BLOCKED. |
| **B. Profil LSM/AppArmor z etykietą endpointu i kontrolą delegacji** | Chroniona polityka wiąże start z profilem; klient dodatkowo sprawdza `SO_PEERSEC`, jeżeli dana konfiguracja Ubuntu je udostępnia. Polityka musi ograniczać exec, zmianę profilu, ptrace, fork oraz użycie/przekazanie FD i ładowanie aplikacji. Sama nazwa profilu nie jest pomiarem kodu. | Bez capabilities klienta i bez nowego brokera; root ładuje/utrzymuje politykę, jądro i LSM stają się jawną częścią dowodu. Kompatybilność parsera/AF_UNIX mediation w Noble wymaga sprawdzenia. | Konkretny alternatywny prototyp z profilem przejścia start → zamknięta usługa. Jeżeli AppArmor nie egzekwuje potrzebnych operacji na przekazanych FD, wariant odpada lub wymaga dodatkowego filtra A. Nie deklarować tej funkcji na podstawie samej obecności AppArmor. |
| **C. Uprzywilejowany obserwator/broker** | Osobny zaufany komponent uzyskuje exe przez ptrace/root i wiąże obserwację z endpointem. Sam raport PID/hash sprzed żądania pozostawia wyścig i problem innego handlera; potrzebuje także egzekwowanej ciągłości albo pośrednictwa w kanale. | Nowy proces root lub `CAP_SYS_PTRACE` (właściwy userns); większy zakres ujawnienia pamięci/sekretów i większa powierzchnia zaufania. Nie przyznawać capability całemu Pythonowi runtime. | Rozważać wyłącznie po jawnej decyzji, że niezależny pomiar exe jest niezbędny. Pozytywny odczyt exe nie kwalifikuje aplikacji Pythona ani efektów SQL. Samo dodanie capability nie spełnia celu. |

### Wymagane doprecyzowanie wariantu A

1. Root chroni dokładny launcher, interpreter, aplikację i komplet zależności;
   podpisy są sprawdzane przed uruchomieniem. Brak środowiska użytkownika,
   pluginów, `.pth`, obcych importów i ścieżek zapisu umożliwiających kod.
   Inwentarz jest dowodem zatwierdzenia, a nie deklaracją pomiaru procesu.
2. Zaufany kod startowy ładuje potrzebne moduły i zamyka niezatwierdzone FD.
   Nie ma wcześniejszych dzieci, obcych listenerów ani niezabezpieczonych wątków.
   Następnie ustawia `PR_SET_NO_NEW_PRIVS`, `PR_SET_DUMPABLE=0` i nieodwracalny
   filtr seccomp dla wszystkich wątków/ABI; niepowodzenie kończy proces.
   Możliwy launcher osadza interpreter, zamiast wykonywać kolejny exec po
   założeniu filtra. Dokładny wybór wymaga review kodu i zamknięcia importów.
3. Filtr blokuje co najmniej `execve`, `execveat`, `fork`, `vfork`, `clone`,
   `clone3`, `sendmsg`, `sendmmsg` oraz drogi io_uring pozwalające ominąć
   kontrolę operacji. Potrzebny jest przejrzany profil dopuszczeń syscalli,
   z odrzuceniem obcych architektur, a nie deklaracja kompletności tej listy.
   Zakaz przekazania FD obejmuje listener i zaakceptowane połączenia.
   Dumpable=0 ogranicza ptrace/pidfd_getfd przez inny proces tego UID;
   root pozostaje zaufany. Nie wolno później cofnąć tej ochrony.
   Zwykły seccomp BPF nie odczytuje struktur pod wskaźnikiem `msg_control`;
   wariant zakłada odmowę całych dróg wysyłania ancillary FD. Polityka musi
   też zabronić zmiany dumpable na 1 i innych operacji cofających zamknięcie.
4. Dopiero po skutecznym zamknięciu powstaje listener, bez socket activation.
   Rodzic nie pozostawia dzieci, które mogły wcześniej odziedziczyć FD.
   Żadna ścieżka obsługi danych nie wykonuje eval/exec/importu spoza
   zatwierdzonej aplikacji. Zakaz syscallu exec nie zamyka sam Pythona.
5. Klient koreluje chroniony kontrakt instalacji, autentyczny systemd,
   MainPID/okres, socket inode i socket-bound pidfd przed pierwszym send.
   Restart, śmierć, brak obserwacji albo rozbieżność oznaczają odmowę.
   `Seccomp: 2`, liczba filtrów, plik „ready” lub odpowiedź serwera nie dowodzą
   dokładnej treści BPF. Podstawą jest zatwierdzony kod startu i wymuszona
   kolejność, co stanowi jawną zmianę dotychczasowej granicy zaufania.

Nie można dodać `SystemCallFilter=~execve execveat` do obecnego unitu i uznać
problemu za rozwiązany: filtr systemd obejmuje także początkowy exec usługi.
Obecne `NoNewPrivileges`, `MemoryDenyWriteExecute`, `ProtectSystem` i
`CapabilityBoundingSet=` nie zakazują późniejszego exec/fork/SCM_RIGHTS.
Jeżeli poprawne działanie wymaga procesów potomnych, przekazywania socketów
lub niezaufanych rozszerzeń, model A trzeba ponownie rozstrzygnąć.

## Ustalenia bezpośrednio z repozytorium

- `freshness_runtime_ipc._require_runtime_identity` wymaga rzeczywistych,
  odrębnych kont NSS i ścisłych grup. `_check_server` porównuje wymagany hash;
  `None` nigdy nie stanowi zatwierdzonego obrazu. W
  `RuntimeFreshnessVerifierClient.verify_exact_candidate` kontrole serwera
  poprzedzają ustawienie `started` i pierwszy `sendall`.
- `linux_service_ipc.capture_process_identity(..., allow_missing_executable=True)`
  pozwala zapisać brak odczytu jako `None`. `PinnedReadOnlyPeer` zachowuje tę
  obserwację, wiąże fdinfo z socket-bound pidfd i ponownie sprawdza życie.
  Nie przyznaje przez to prawa do przyjęcia tożsamości aplikacji.
- `linux_physical_ipc._pinned_verifier_socket` chroni ancestry/inode/tryby
  AF_UNIX. `ExpectedPeerProcess` jest zaufanym wejściem z niezależnej
  obserwacji, nigdy dowodem pochodzącym z odpowiedzi weryfikowanego serwera.
- `freshness_service_identity.qualify_freshness_service_identity` nie ma
  jawnego warunku UID=0, lecz wymaga odczytu exe, `_program_observation`,
  podpisanego runtime, dokładnego D-Bus ExecStart/ExecStartPre oraz pidfd.
  Wykonanie tej obserwacji jako root nie dowodzi dostępności dla runtime.
- `production_local_installer._trusted_bootstrap` i `install`,
  `freshness_runtime_bootstrap.main` oraz produkcyjny unit
  wiążą zatwierdzone pliki i kolejność startu. Instalator nie nadaje klientowi
  prawa ptrace i nie zapewnia ciągłości procesu po ExecStartPre.
- Istniejące testy klienta używają rzeczywistych AF_UNIX/PIDFD, ale modelują
  konta, chronioną instalację i cgroup. Writer jest modelem. Nie są dowodem
  dwóch UID, zainstalowanego systemd ani rzeczywistych odmów PostgreSQL.

Codebase Memory sprawdzono przed analizą zależności: brak narzędzi serwera,
zasobów MCP i lokalnego `codebase-memory-mcp`; wyszukiwanie integracji nie
udostępniło tego serwera. Nie wykonano zapytań grafu, odświeżenia ani
`check_index_coverage`. To jawna luka przygotowania; powyższe ustalenia
zweryfikowano bezpośrednio w źródłach, bez deklaracji kompletności indeksu.

## Najmniejsza kwalifikacja natywna i dalsza akceptacja

[Harness i instrukcja](../runbooks/stage9_cross_uid_native_identity.md)
obejmują oddzielne UID/GID, systemd, chronioną ścieżkę, AF_UNIX,
SO_PEERCRED/SO_PEERPIDFD oraz odmowę niezmienionego klienta przy rzeczywistym
braku exe. Obserwacyjny ślad Pythona wskazuje rzeczywistą kontrolę i wyjątek
w wywołaniu klienta; nie zastępuje ani nie zmienia kontroli tożsamości.
Listener tylko liczy bajty i nie udostępnia przygotowania ani SQL.
Wynik potwierdza wyłącznie odmowę transportu przed wysłaniem; nie zastępuje
dowodu braku zmian w rzeczywistym PostgreSQL. Wcześniejsza odmowa innego
warunku, niedostępny pidfd lub czytelne exe nie zaliczają tego przypadku.

W tej sesji nie udostępniono zatwierdzonej Ubuntu VM. Środowisko robocze
Debian z PID 1 innym niż systemd nie kwalifikuje próby. Nie tworzyć w nim
kont, usług ani klastra. Uruchomienie administracyjne wymaga odrębnej zgody
na konkretną jednorazową VM; dokument ani wynik lokalnych testów jej nie daje.

Po akceptacji architektury osobny prototyp musi objąć: udany start i właściwą
wymianę, execve/execveat innego i tego samego interpretera, fork/vfork/clone,
listener odziedziczony przed zamknięciem, SCM_RIGHTS listenera i połączenia,
io_uring, ptrace/pidfd_getfd, restart/śmierć podczas wymiany, podmianę socketu,
brak D-Bus/procfs/pidfd oraz zmianę inwentarza/importów. Każda zakazana
operacja wymaga rzeczywistej odmowy jądra albo jednoznacznego BLOCKED.
Próba nie może uznać samej odmowy klienta z innej przyczyny za skuteczność
filtra. Zaufany launcher musi mieć także test odmowy instalacji filtra przed
listen. Nie wystarczy test po fakcie, który wykrywa exec po wysłaniu danych.

Pełny dowód usługi i efektów wymaga osobnego
[kontraktu ACCEPT/REJECT VM](../runbooks/STAGE9_FRESHNESS_SERVICE_VM_ACCEPTANCE.md)
i [harnessu v2](../runbooks/STAGE9_LINUX_VM_HARNESS.md), rzeczywistego
PostgreSQL oraz niezależnie zatwierdzonego materiału TEST_ONLY. Nie tworzyć
syntetycznej zgody PRODUCTION ani nowego podpisu wyłącznie w celu odblokowania
próby. Brak takich wejść pozostawia BLOCKED/NOT_EXECUTED.

## Źródła ograniczeń platformy

Zweryfikowano źródła bazowego Linux 6.8 i systemd 255 odpowiadających rodzinie
Ubuntu 24.04; dokładną wersję/backporty uruchomionej VM należy zachować w raporcie.

- Linux: [init_peercred, copy_peercred i unix_listen](https://github.com/torvalds/linux/blob/v6.8/net/unix/af_unix.c#L686),
  [unix_stream_connect](https://github.com/torvalds/linux/blob/v6.8/net/unix/af_unix.c#L1600),
  [SO_PEERCRED i SO_PEERPIDFD](https://github.com/torvalds/linux/blob/v6.8/net/core/sock.c#L1750).
- Procfs: [kontrola dostępu do linków procesu](https://github.com/torvalds/linux/blob/v6.8/fs/proc/base.c#L674),
  [warunki ptrace](https://github.com/torvalds/linux/blob/v6.8/kernel/ptrace.c#L276),
  [starttime w do_task_stat](https://github.com/torvalds/linux/blob/v6.8/fs/proc/array.c#L586).
- Systemd: [odczytowe Properties.Get/GetAll](https://github.com/systemd/systemd/blob/v255/src/core/org.freedesktop.systemd1.conf#L29),
  [semantyka ExecStart i MainPID](https://github.com/systemd/systemd/blob/v255/man/org.freedesktop.systemd1.xml#L4627),
  [ograniczenie początkowego exec przy filtrowaniu](https://github.com/systemd/systemd/blob/v255/man/systemd.exec.xml#L2428).
- [Seccomp BPF: no_new_privs, filtry i ograniczenia inspekcji argumentów](https://docs.kernel.org/userspace-api/seccomp_filter.html).

## Nierozwiązane blokady

Pozostają `RUNTIME_IPC_APPLICATION_ADAPTER_UNAVAILABLE`,
`ACTUAL_RUNTIME_CLIENT_EXCHANGE_UNPROVEN`,
`RUNTIME_UNAUTHORIZED_IDENTITY_DENIAL_UNPROVEN` oraz
`DURABLE_CHALLENGE_REPLAY_PROTECTION_UNAVAILABLE`. Nie podłącza się
AccountGenesis. Jego rozbieżność z zamrożonym `REVIEWED_CODE_DIGESTS`
pozostaje aktywna; piny, digesty, podpisy, manifesty, progi CI i
`deployment/stage9_current_status.json` pozostają bez zmian.
Nie ma zgody na merge ani deklarację gotowości Stage 9.
