# Stage 9 — macierz prób hybrydy AppArmor i seccomp TEST_ONLY

Baza: `main` po PR #3252, commit
`5cb848d656c2b7d9d8b06e654be7911f3c9fac0f`, 5 listopada 2026 r.
To **plan kwalifikacji**, nie zapis wykonanych odmów jądra.
Wszystkie poniższe próby wymagające uruchomienia jednostki, polityki lub
zmiany stanu AppArmor mają obecnie **NOT_RUN**. Bez osobnego zatwierdzenia
konkretnej jednorazowej VM ich wykonanie ma status **BLOCKED**.
Stage 9 pozostaje **IN_PROGRESS / PRODUCTION_NO_GO**.

Wnioski o platformie opisuje
[przegląd źródeł](../architecture/stage9_hybrid_platform_review.md).
Poniższy plan nie rozszerza podstawy zaufania klienta, nie zastępuje
zatwierdzenia kodu ani nie zezwala na produkcyjną integrację.

## Rodzaje dowodów

| Rodzaj | Co może potwierdzić | Czego nie potwierdza |
| --- | --- | --- |
| Test syntetyczny istniejącego `stage9_verifier_seal_model.py` | Decyzję interpretera cBPF i kolejność deklarowanych etapów | Instalacji filtra, działania LSM, braku wcześniejszego tracera lub mapowań |
| Statyczna kontrola AppArmor | Przyjęcie składni przez dokładny parser dla jawnego zestawu cech, bez ładowania | Aktywacji profilu, trybu enforce, zgodności z uruchomionym kernelem lub odmowy syscalla |
| Rzeczywista odmowa jądra | Konkretną operację w konkretnej konfiguracji, po przypisaniu odmowy do mechanizmu | Ogólnego zamknięcia Pythona i bezpieczeństwa innego kernela lub wydania |
| Test niewykonany | Wyłącznie istnienie planu | Żadnej właściwości wykonania; wynik pozostaje NOT_RUN |

Wynik `EPERM`/`EACCES`, `SIGSYS`, `aa-status`, `Seccomp: 2`, nazwa profilu,
`SO_PEERSEC` lub wpis `AppArmorProfile=` samodzielnie nie zalicza przypadku.
PASS wymaga także dowodu, że test dotarł do badanej operacji. Nie wolno
zaliczyć wcześniejszej odmowy DAC, Yama, systemd, innego seccomp albo odmowy
produkcyjnego klienta jako dowodu działania projektowanego ograniczenia.

## Odtworzenie kontroli dostępnych bez zmiany hosta

Istniejący model i nową ograniczoną sondę można uruchomić oddzielnie:

```console
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 python -m pytest tests/scripts/test_stage9_verifier_seal_model.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 python -m pytest tests/security/test_stage9_hybrid_seccomp_probe.py -q
```

[`stage9_hybrid_seccomp_probe.c`](../../scripts/diagnostics/stage9_hybrid_seccomp_probe.c)
to statyczny program Linux x86-64, który instaluje dumpable=0, NNP i własny
filtr w izolowanym potomku. Testy kompilują dwa warianty różniące się errno
odmowy, sprawdzają rzeczywiste wyniki zakazanych syscalli, zabicie x32 oraz
odmowę przy błędzie instalacji filtra. Stanowi to osobną kategorię dowodu
**NATIVE_SECCOMP_CHILD**. Faktyczny wynik wykonania, wersja kompilatora
i kernel należą do raportu sesji; brak kompilatora lub platformy to NOT_RUN.

Sonda używa nieprawidłowych argumentów i nie tworzy listenera ani połączenia,
nie przekazuje rzeczywistego FD i nie uruchamia interpretera. Zmienione errno
przypisuje odmowy badanemu filtrowi, ale nie dowodzi kompletności filtra
aplikacji, rzeczywistej kontroli SCM_RIGHTS, ochrony fazy A, braku wcześniejszych
mapowań ani tożsamości AF_UNIX. Wariant błędu instalacji dowodzi przerwania
tej sondy przed próbami; nie kwalifikuje kolejności seal → listen przyszłego
launchera, którego sonda nie implementuje.

## Przygotowanie jednorazowej VM

Przed osobnym wykonaniem operator musi wskazać konkretną VM Ubuntu Server
24.04, dokładny kernel i pakiety systemd/AppArmor, a niezależny recenzent
zatwierdzić materiały TEST_ONLY oraz zakres zmian hosta. Samo oznaczenie
dystrybucji jako 24.04 nie rozstrzyga wersji GA/HWE ani backportów.
Kopie z wyłączonym AppArmor, complain mode i kontrolnymi politykami są
odrębnymi wariantami jednorazowej VM; nigdy nie zmieniać w ten sposób hosta
produkcyjnego lub współdzielonego.

Materiał próby obejmuje:

1. Chroniony, niezależnie przejrzany kod launchera, jego ELF/loader/biblioteki,
   profil, jednostkę i dokładne bajty filtra. Żadnych produkcyjnych kluczy,
   dokumentów, PostgreSQL ani AccountGenesis. Jedyny protokół to stały,
   ograniczony komunikat testowy.
2. Osobne konta TEST_ONLY klienta, celu i napastnika. Napastnik tego samego
   UID co cel musi być osobnym, początkowo nieograniczonym procesem, także
   poza cgroup jednostki. Nie dodawać capabilities, setuid ani brokera root.
   Istniejący administrator VM służy wyłącznie zatwierdzonej konfiguracji
   i niezależnej obserwacji.
3. Chronione katalogi kodu i endpointu, stały argv/cwd/środowisko, wyłączone
   socket activation i dziedziczenie kanałów systemd. Zinwentaryzować FD
   **łącznie z 0/1/2**, dzieci, wątki, mapowania, ringi i wejścia środowiska.
   Brak możliwości wykluczenia wcześniejszego posiadacza FD lub mapowania
   oznacza BLOCKED, nie „prawdopodobnie czysto”.
4. Faza A musi objąć rzeczywistą granicę od startu, w tym przejście systemd
   do UID celu przed `execve`. Faza B obejmuje potwierdzone dumpable=0, NNP,
   dokładny seccomp dla właściwego ABI/TSYNC i dopiero potem socket/bind/listen.
   Listener nie może powstać przy błędzie dowolnego wcześniejszego kroku.
5. Niezależny zapis boot ID, wersji pakietów i kernela, aktywnych LSM,
   kompletu cech AppArmor, stanu enforce, efektywnego unitu/drop-inów,
   UID/GID/grup, cgroup, osi czasu i wyników operacji. Zrzut aktywnej polityki
   i pochodzenie filtra muszą być związane z zatwierdzonymi materiałami;
   oświadczenie procesu ani własny plik ready nie są taką kotwicą.

Odczytowe dane wersji można zebrać następująco; brak dostępu do wymaganych
danych jest wynikiem BLOCKED, nie powodem do nadania uprawnień klientowi:

```console
uname -a
cat /etc/os-release
systemd --version
apparmor_parser --version
dpkg-query -W systemd apparmor apparmor-utils linux-image-generic
cat /sys/kernel/security/lsm
cat /sys/module/apparmor/parameters/enabled
```

Należy zachować również dokładną wersję uruchomionego pakietu kernela oraz
pełny zestaw cech z securityfs. Polecenie dotyczące metapakietu
`linux-image-generic` nie potwierdza, z którego obrazu uruchomiono VM.

## Przypisanie odmowy do ochrony

Dla każdej operacji zapisać PID/TID, etap A/B, ABI, numer syscalla, istotne
argumenty, wynik, errno/sygnał i skutek uboczny. Sekretne dane nie są potrzebne.
Powiązać odmowę AppArmor z niezależnym rekordem audytu dla właściwego PID,
profilu, operacji i czasu; sam brak rekordu nie potwierdza ani nie wyklucza
odmowy. Dla seccomp ustalić dokładny filtr i jego działanie różnicowo.

W każdej niezależnej kopii VM utrzymać pozostałe mechanizmy identyczne:

- **A:** badany profil enforce, bez badanego filtra B — rozstrzyga odmowy LSM;
- **B:** kontrolny profil, dokładny badany filtr — rozstrzyga odmowy seccomp;
- **A+B:** właściwa hybryda — sprawdza współdziałanie;
- **kontrola:** bez badanego ograniczenia, z tym samym UID/DAC/Yama/ABI —
  musi wykazać, że operacja jest osiągalna i helper poprawny.

Kontrole są wyłącznie materiałem laboratoryjnym. Jeżeli kontrola też odmawia,
wynik jest **NOT_PROVEN** do ustalenia przyczyny. Nie osłabiać konfiguracji
rzeczywistej usługi. Nie instalować `strace` jako tracera celu w próbie,
która ma dowodzić zakazu ptrace; instrumentacja nie może tworzyć badanej
luki. Korzystać z niezależnej obserwacji oraz jawnych barier w osobnym
diagnostycznym buildzie i odróżnić go od docelowego artefaktu.

## Macierz podstawowa

W kolumnie „Próba” podano konkretny bodziec do przyszłego helpera natywnego.
To nie są deklaracje wykonania. Każda pozycja ma obecnie **NOT_RUN**;
jej wymagany rezultat jest warunkiem zaliczenia kwalifikacji.

| Scenariusz | Próba i obserwacja | Wymagany rezultat |
| --- | --- | --- |
| AppArmor wyłączony | Osobna VM uruchomiona bez aktywnego AppArmor; spróbować przejść bramkę przed startem celu. Sprawdzić także, że samo `AppArmorProfile=` nie zastępuje bramki. | **BLOCKED przed startem próby**; brak procesu aplikacji i listenera. Pominięta dyrektywa lub działająca aplikacja oznacza REJECT. |
| Profil niezaładowany | W kopii VM bez badanego profilu zachować tę samą jednostkę bez prefiksu `-`; zapisać błąd managera oraz niezależnie brak aplikacji/socketu. | **BLOCKED**; żadnego fallbacku do profilu nieograniczonego. |
| Profil w complain mode | Tę samą nazwę profilu ustawić w complain tylko w zatwierdzonej kopii VM; użyć tego samego wejścia startowego. | **BLOCKED** przed aplikacją. Zgodna nazwa/etykieta nie wystarcza. |
| Niezatwierdzony exec | W fazie A uruchomić obcy ELF, ponowny exec zatwierdzonego launchera i wykonanie przez FD; po B wywołać `execve`, `execveat(AT_EMPTY_PATH)` i warianty ABI, także dla tego samego interpretera. | Odmowa wszystkich niezatwierdzonych przejść w A i **każdego późniejszego exec w B**. Zapisać, czy odmówił profil, filtr czy inny mechanizm. |
| Ptrace tego samego UID przed inicjalizacją | Napastnik spoza cgroup atakuje proces systemd po zmianie credentials, przed wykonaniem docelowego obrazu, w loaderze i przed wczesnym dumpable=0. Badać `PTRACE_ATTACH` i `PTRACE_SEIZE`; osobno rozpocząć śledzenie przed przejściem do docelowego profilu. W barierowym buildzie próbować zmiany rejestrów/pamięci i kontynuacji celu. | Każda droga wpływu musi zostać odrzucona **przed pierwszą możliwą modyfikacją**. Wcześniejszy tracer, który zdążył zmienić stan, daje **REJECT architektury**, nawet gdy później zostaje odcięty. Jednorazowe nieudane trafienie wyścigu daje NOT_PROVEN. |
| Ptrace po inicjalizacji | Po B ten sam napastnik próbuje attach/seize, `process_vm_readv`, `process_vm_writev`, otwarcia `/proc/PID/mem` oraz `pidfd_getfd` listenera i połączenia. Testy odczytu i zapisu rejestrować osobno. | Odmowa przed dostępem lub skopiowaniem FD; niezależne przypisanie do dumpable/LSM. Filtr seccomp celu nie filtruje syscalli napastnika. |
| Podmiana biblioteki lub modułu Python | Jako UID celu próbować zapisu/rename/symlink do każdego elementu zatwierdzonego closure i jego rodziców. Osobno obcy `LD_PRELOAD`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `.pth`, user-site, cwd, zastany `.pyc` oraz plikowy import z obcej ścieżki przed i po B. | Brak możliwości podmiany i wykonania obcych bajtów. Niedozwolony plik ma zostać odrzucony lub ignorowany przed wykonaniem; brak pełnego zamknięcia zależności daje NOT_PROVEN. Wykluczyć wykonywanie kodu przed jego weryfikacją. |
| SCM_RIGHTS listenera i połączenia | Próbować `sendmsg` i `sendmmsg` z `SCM_RIGHTS` osobno dla listenera i zaakceptowanego socketu; odbiorcą jest ten sam UID poza profilem oraz proces w tym samym profilu. Powtórzyć kierunek odbioru przez `recvmsg`/`recvmmsg` i sprawdzić rzeczywistą tablicę FD odbiorcy. | Odmowa bez nowego posiadacza FD. W B zakazać całych niepotrzebnych syscalli; filtrowanie wskaźnika `msg_control` nie wystarcza. Poświadczenia klienta mogą nadal wskazywać dawnego właściciela. |
| Fork/clone i odziedziczone FD | Po B: `fork`, `vfork`, `clone`, `clone3`, także tworzenie wątku. Przed B: próba potomka lub współdzielenia `CLONE_FILES`/`CLONE_VM`, odziedziczone listener/połączenie/ring i socket w FD 0/1/2. | Odmowa nowych procesów/wątków po B. Wcześniejsze dzieci, współdzielenie i niezatwierdzone FD powodują odmowę startu przed listenerem; pusta lista FD w celu nie dowodzi braku obcych kopii. |
| io_uring i wcześniejsze mapowania | Osobno `io_uring_setup`, `io_uring_register`, `io_uring_enter`; kontrolować wysyłanie ancillary FD z SQE. W wariancie celowo skażonym przed B utworzyć ring, mapowania SQ/CQ i SQPOLL, zamknąć sam FD, a potem próbować wykonywania zleceń. Osobno odziedziczone zapisywalne mapowanie współdzielone. | Nowe drogi po B mają odmówić. Zaufany start musi uniemożliwiać wcześniejsze ringi/mapowania/operacje; możliwość skutku po B pomimo zamknięcia FD daje **REJECT**. Nie przypisywać AppArmor 6.8 nowszych hooków tylko dlatego, że parser rozpoznaje składnię. |
| Nieudane ustanowienie seccomp | W osobnym buildzie TEST_ONLY wymusić błąd ustawienia dumpable, NNP, nieprawidłowy filtr, błąd TSYNC lub dodatni TID zwrócony przy niesynchronizowanym wątku. Obserwować zakończenie i brak socket/bind/listen na wszystkich odnogach błędu. | Zakończenie przed listenerem, bez retry/fallbacku. Sam `Seccomp: 2` po odziedziczonym filtrze nie zalicza sukcesu badanego filtra. |
| Prawidłowa minimalna komunikacja AF_UNIX | Dopiero po zaliczeniu bramek wykonać jeden stały komunikat między odrębnymi UID. Powiązać chroniony inode socketu, `SO_PEERCRED`, socket-bound `SO_PEERPIDFD`, życie procesu, cgroup, boot ID, MainPID/ExecMainPID i okres systemd przed/po. Nie wywoływać klienta produkcyjnego z poluzowanym hashem. | Dodatni wynik **wyłącznie przy pełnych dowodach A+B i braku delegacji**. Jest dowodem ograniczonego demo, bez weryfikacji dokumentu, SQL i kwalifikacji produkcyjnej. Brak odczytu/pola albo rozbieżność oznacza odmowę. |

## Uzupełnienia konieczne do zamknięcia granicy

- **Wcześniej otwarte `/proc/PID/mem`:** napastnik tego samego UID otwiera
  uchwyt w przedziale przed ograniczeniami. Po przejściu profilu i dumpable=0
  próbuje użyć zachowanego FD. Trzeba zbadać także exec celu i żywotność
  starego `mm`; nie zakładać, że późniejsza kontrola dostępu cofa wcześniejszy
  dostęp. Wynik musi rozróżniać unieważniony obraz od faktycznej blokady LSM.
- **Zmiana dumpable/polityki:** po B spróbować `prctl(PR_SET_DUMPABLE, 1)`,
  zmiany profilu/hat, `seccomp`, `personality`, namespace i credentials.
  Wszystkie niepotrzebne drogi muszą odmawiać. NNP nie zabrania samodzielnie
  exec ani podniesienia dumpable.
- **Kod Python w pamięci:** w kontrolowanym, odrębnym kandydacie wykonać
  obiekt kodu lub interpretować kod pochodzący z danych już w pamięci po B,
  bez nowego pliku, exec systemowego i pamięci PROT_EXEC. Jeśli jest to
  możliwe, wymóg zakazu niezatwierdzonego kodu Python ma **REJECT**.
  Usunięcie `eval`, zmiana `sys.path` lub hook audytu nie dowodzą nieodwracalnej
  granicy. Natywne demo bez interpretera tego przypadku nie kwalifikuje.
- **Sygnały:** ten sam UID poza profilem próbuje sygnałów wpływających na cel
  od fazy A, nie tylko po B; osobno wymagany zaufany stop systemd.
  Zatrzymanie/śmierć musi kończyć próbę odmową. Nie uznawać cgroup za ochronę
  przed obcym procesem tego samego UID.
- **Restart/śmierć/podmiana socketu:** zmieniać po jednym fakcie przed send
  i podczas wymiany. Po pierwszym send niejednoznaczność pozostaje
  `OUTCOME_UNKNOWN`, bez retry; późniejsza odmowa nie cofa wysłanych danych.
- **ABI:** obce architektury, i386/socketcall, x32 i nieznany syscall mają
  odmówić. Rzeczywiste argumenty glibc trzeba sprawdzić, zamiast zakładać
  zgodność z 64-bitowym syntetycznym `seccomp_data`.

## Powiązanie z istniejącym verifierem i decyzje

Źródłem obecnego kontraktu są bezpośrednio przejrzane pliki bazowego commitu:

- [`freshness_runtime_ipc._check_server`](../../bot_core/freshness_runtime_ipc.py)
  wymaga zgodnego `exe_sha256`; `None` nie daje zgody. Kontrole w
  `verify_exact_candidate` poprzedzają pierwszy `sendall`, a późny błąd daje
  `OUTCOME_UNKNOWN` bez automatycznego retry.
- [`PinnedReadOnlyPeer`](../../bot_core/linux_service_ipc.py) bierze PIDFD
  z faktycznego socketu. `allow_missing_executable=True` zapisuje brak obrazu,
  ale nie ustanawia nowej podstawy akceptacji. PIDFD, cgroup i zapisane
  poświadczenia nie identyfikują późniejszego posiadacza przekazanego FD.
- [`freshness_service_identity`](../../bot_core/freshness_service_identity.py)
  wymaga zatwierdzonego runtime, dokładnego argv/efektywnego unitu,
  bootstrapu i bieżącego obrazu. Natywny launcher i profil nie spełniają tego
  kontraktu przez samo dopisanie pola AppArmor.
- [`production_local_installer`](../../deployment/production_local_installer.py)
  publikuje po pełnym preflight i weryfikuje dokładny unit/runtime, bez mutacji
  usług/NSS/DB. Przyszła publikacja/ładowanie polityki i obsługa aktualizacji
  wymagałyby nowego, osobno przejrzanego zakresu; ten plan go nie implementuje.
- [`freshness_runtime_bootstrap`](../../deployment/freshness_runtime_bootstrap.py)
  używa OpenSSL przez subprocess i przekazuje memfd; nie można bezpośrednio
  wykonywać go za filtrem zabraniającym procesów/exec.
- [`freshness_verifier_entrypoint`](../../deployment/freshness_verifier_entrypoint.py)
  weryfikuje aplikację/interpreter przed importem usługi, ale nie ustanawia
  seal; [`freshness_verifier_service.main`](../../bot_core/freshness_verifier_service.py)
  łączy się z PostgreSQL przed wystawieniem socketu. Nie jest wejściem demo.
- [Produkcyjny unit](../../deployment/systemd/cryptohunter-freshness-verifier.service)
  ma NNP i ograniczenia systemd, ale nie badaną dwufazową granicę.
  Nie wolno podmieniać go jednostką demonstracyjną.

Obecnie **REJECT** dotyczy pełnego wymagania ciągłości interpretowanego kodu
przy możliwości niezatwierdzonego wykonania w pamięci. Odporność startu na
same UID, rzeczywiste odmowy A+B i odporność na delegację pozostają
**NOT_PROVEN**, dopóki nie ma dowodu źródłowego oraz kwalifikacji dokładnego
artefaktu na zatwierdzonej platformie. **ACCEPT zakresu TEST_ONLY** oznacza
wyłącznie zachowanie izolacji od produkcyjnej usługi i jej danych; nie jest
ACCEPT bezpieczeństwa proponowanego verifiera.

Minimalny następny krok to decyzja, czy wymagana jest odporność na dowolny
kod Python w pamięci, oraz niezależne review ochrony przed pierwszym exec.
Jeżeli wymagania pozostają bez zmian, potrzebna jest istotna zmiana
architektury, np. ograniczony natywny proces weryfikujący z przejrzanym
parserem i domkniętym kodem, zamiast obejścia odmowy klienta. Dopiero potem
ma sens osobne zatwierdzenie jednej VM i małej próby wczesnego ptrace.
Nie wykonywać merge, nie zmieniać podpisów/digestów/pinów ani
`deployment/stage9_current_status.json` i nie podłączać AccountGenesis.
