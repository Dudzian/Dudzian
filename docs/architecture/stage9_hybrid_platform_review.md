# Stage 9 — przegląd platformy hybrydy AppArmor i późnego seccomp

Stan: 5 listopada 2026 r. Baza repozytorium:
`5cb848d656c2b7d9d8b06e654be7911f3c9fac0f`, po PR #3252.
**REJECT gwarancji pełnej ochrony hybrydy w zadanym modelu napastnika.**
Przegląd źródeł wykazał konkretną granicę ochrony wobec procesu
`unconfined` tego samego UID. Nie jest to raport wykonanego exploitu ani
kwalifikacja Ubuntu. Stage 9 pozostaje **IN_PROGRESS / PRODUCTION_NO_GO**.

## Źródła i odtworzenie przeglądu

Przeczytano rzeczywiste źródła systemd, parsera i dystrybucyjnego jądra.
Codebase Memory nie jest dostępne w aktywnym katalogu narzędzi; nie wykonano
zapytania grafu ani `check_index_coverage`. Repozytorium pozostaje źródłem
prawdy dla integracji. Poniższe sumy identyfikują odczytane źródła zewnętrzne;
nie są nowymi pinami instalacji lub kwalifikacji produktu.

| Materiał | Dokładny zakres i identyfikator |
| --- | --- |
| systemd | Tag `v255`, commit `db11bab38ccf1ed257f310d29070843d4c58ea01`; `src/core/exec-invoke.c`, `src/shared/apparmor-util.c`, `man/systemd.exec.xml`. Nie odtworzono całego pakietu systemd Ubuntu z jego poprawkami. |
| Ubuntu GA, wydanie początkowe | `linux_6.8.0-31.31`: archiwum bazowe oraz dystrybucyjny diff zastosowany do plików AppArmor i istotnych plików jądra. |
| Ubuntu GA, dodatkowy aktualny materiał | `linux_6.8.0-149.149`, opublikowany 26 października 2026 r.; to samo archiwum bazowe i jego osobny diff. Odtworzono i przeczytano istotne zmiany AppArmor, exec, ptrace, PIDFD, process_vm i io_uring. Nie jest to badanie każdego kernela Noble. |
| AppArmor | Źródła `apparmor_4.0.1really4.0.1.orig.tar.gz`, SHA-256 `b0d72cedc48e533d189ea415bde721ad597101c77fa398fdd2858ec4f58f7e26`; `parser/apparmor.d.pod`, dokumentacja parsera i `aa_change_onexec`. |
| Punkt porównania upstream | Linux `v6.8`, commit `e8f897f4afef0031fe618a8e94127a0934896aba`; nie zastępuje zastosowanych poprawek Ubuntu. |

Źródła do odtworzenia:

- [systemd v255: kolejność uruchomienia](https://github.com/systemd/systemd/blob/db11bab38ccf1ed257f310d29070843d4c58ea01/src/core/exec-invoke.c)
  i [wykrywanie AppArmor](https://github.com/systemd/systemd/blob/db11bab38ccf1ed257f310d29070843d4c58ea01/src/shared/apparmor-util.c).
- [Kontrakt AppArmorProfile= w systemd v255](https://github.com/systemd/systemd/blob/db11bab38ccf1ed257f310d29070843d4c58ea01/man/systemd.exec.xml).
- [Ubuntu linux_6.8.0-31.31.dsc](https://archive.ubuntu.com/ubuntu/pool/main/l/linux/linux_6.8.0-31.31.dsc)
  i [Ubuntu linux_6.8.0-149.149.dsc](https://archive.ubuntu.com/ubuntu/pool/main/l/linux/linux_6.8.0-149.149.dsc).
- [Archiwum bazowe jądra](https://archive.ubuntu.com/ubuntu/pool/main/l/linux/linux_6.8.0.orig.tar.gz),
  SHA-256 `26512115972bdf017a4ac826cc7d3e9b0ba397d4f85cd330e4e4ff54c78061c8`,
  230060117 bajtów.
- [Diff Ubuntu -31.31](https://archive.ubuntu.com/ubuntu/pool/main/l/linux/linux_6.8.0-31.31.diff.gz),
  SHA-256 `21ba797c2f212456ded0de09e5d016a857a070d43513522df5d1b414ca94de04`,
  1162846 bajtów.
- [Diff Ubuntu -149.149](https://archive.ubuntu.com/ubuntu/pool/main/l/linux/linux_6.8.0-149.149.diff.gz),
  SHA-256 `b445915e8310e892e2f2f6ea4fc451e3602e7fedd64c590f1785f7a684a4af51`,
  11931455 bajtów.
- [Źródła AppArmor 4.0.1](https://archive.ubuntu.com/ubuntu/pool/main/a/apparmor/apparmor_4.0.1really4.0.1.orig.tar.gz)
  i [metadane pakietu Noble](https://archive.ubuntu.com/ubuntu/pool/main/a/apparmor/apparmor_4.0.1really4.0.1-0ubuntu0.24.04.9.dsc).

Sumy obu materiałów jądra porównano z `Checksums-Sha256` właściwych `.dsc`.
Odtworzenie: rozpakować archiwum bazowe, zastosować odpowiedni diff z `patch -p1`,
a następnie odczytać wskazane niżej funkcje. W tej pracy zastosowano
wyodrębnione sekcje diffów do pełnego poddrzewa `security/apparmor/` i wybranych
plików wywołujących; nie kompilowano jądra ani całego drzewa. Nie weryfikowano
niezależnie podpisu OpenPGP `.dsc`. HTTPS i zgodność hashy nie kwalifikują
uruchomionego kernela lub konfiguracji konkretnego hosta.

HWE, inne architektury, dodatkowe backporty, dokładny pakiet systemd Noble,
włączone LSM i ich kolejność wymagają osobnego przypisania do VM. Host tej
sesji zgłasza kernel `6.18.57`; jego zachowanie nie jest dowodem dla Noble GA.
Nie wykonywano tu uprzywilejowanych prób, ładowania profili ani startu usługi.

## Aktywacja profilu i granica exec

**ACCEPT mechanizmu, NOT_PROVEN konkretnego startu.** Systemd v255 używa
`aa_change_onexec(context->apparmor_profile)` przed wykonaniem programu,
a nie zmiany profilu dopiero wewnątrz aplikacji. W odczytanym źródle
`enforce_user()` i `setresuid()` poprzedzają ten krok. Wywołanie
`aa_change_onexec` przygotowuje zmianę; samo nie obejmuje natychmiast
wszystkich wcześniejszych czynności procesu wykonawczego systemd.

Jądro realizuje zmianę w `apparmor_bprm_creds_for_exec()`
(`security/apparmor/domain.c`) i zatwierdza credentials w `begin_new_exec()`
(`fs/exec.c`). `fs/binfmt_elf.c` wykonuje `begin_new_exec()` przed
`load_elf_interp()` i końcowym `START_THREAD`. Nowa etykieta obowiązuje więc
przed wykonaniem instrukcji dynamicznego loadera w userspace oraz przed
pierwszą instrukcją aplikacji. Nie oznacza to, że cała wcześniejsza obsługa
ELF w jądrze przebiega już z etykietą docelową. Zaufanie do loadera,
mapowanych bibliotek i środowiska nadal wymaga zatwierdzonego inwentarza.

Alternatywą jest attachment profilu do dokładnej ścieżki natywnego ELF.
Nie wolno stosować ogólnego attachment do `/usr/bin/python3`, dowolnego
interpretera z PATH lub zapisywalnej ścieżki. Attachment identyfikuje nazwę
pliku, nie mierzy jego zatwierdzonych bajtów. Potrzebna pozostaje kontrola
właściciela, całego łańcucha katalogów, linków, mountów i aktualizacji.
Nie stosuje się przejść `ux`/`Ux`, fallbacku do unconfined ani dowolnego
`change_profile`. Dla docelowego statycznego ELF można całkowicie zabronić
późniejszego exec; dla inicjalizacji dynamicznego Pythona nie jest to dowód
zamknięcia kodu w pamięci.

## Dlaczego sam profil nie zamyka napastnika tego samego UID

**REJECT gwarancji ochrony startu wynikającej wyłącznie z profilu ofiary
oraz późnego dumpable/seccomp.** Kluczowy jest dystrybucyjny kod ptrace,
a nie nazwa profilu lub podręcznikowa lista uprawnień.

W Ubuntu `6.8.0-149.149`, `security/apparmor/task.c`,
`profile_tracee_perm()` zawiera następującą odmowę mediacji:

```c
if (profile_unconfined(tracee) || unconfined(tracer) ||
    !ANY_RULE_MEDIATES(&tracee->rules, AA_CLASS_PTRACE))
        return 0;
```

W tej samej wersji `profile_tracer_perm()` zwraca 0 dla
`profile_unconfined(tracer)`. Profil ofiary z `deny ptrace`, również
`readby` i `tracedby`, nie wymusza więc samodzielnie odmowy wobec atakującego
`unconfined`. W początkowym `6.8.0-31.31` równoważna ścieżka używa
`profile_mediates()` / `label_mediates()`; zwykłe unconfined nie mediuje
klasy ptrace (`policy.c`, `unconfined_mediates`). Sprawdzono oba odtworzone
warianty zamiast ekstrapolować współczesny upstream.

Są trzy odrębne odcinki startu:

1. **Przed exec.** Zaufany systemowy manager uruchomiony jako root zmienia UID.
   `kernel/cred.c:commit_creds()` resetuje dumpable do `fs.suid_dumpable`
   przy zmianie efektywnego UID/GID/fsuid/fsgid i publikuje zmianę z barierą.
   `kernel/ptrace.c:__ptrace_may_access()` ma odpowiadającą barierę i
   sprawdzenie dumpability. Przy `fs.suid_dumpable=0` lub 2 sam fakt, że
   `setresuid()` wystąpił przed `aa_change_onexec()`, nie dowodzi okna attach.
   Start z managera użytkownika lub wcześniej śledzonego procesu jest innym
   kontraktem i nie otrzymuje tego uzasadnienia.
2. **Wewnątrz exec.** `cred_guard_mutex` chroni obliczenie credentials przed
   nowym `ptrace_attach`; `exec_update_lock` synchronizuje wymianę mm i
   odczyty używane przez process_vm/PIDFD. Nie należy przedstawiać chwilowego
   ustawienia dumpable przed zatwierdzeniem etykiety jako samodzielnej,
   udowodnionej wygrywalnej race — te blokady są częścią analizy.
3. **Po exec, przed dumpable=0 w launcherze.** Zwykły exec czytelnego ELF bez
   zmiany credentials ustawia `SUID_DUMP_USER`, zwykle 1 (`fs/exec.c`). Po
   zwolnieniu blokad działa loader i kod przed pierwszym `prctl` launchera.
   Wobec dopuszczonego przez DAC/Yama procesu `unconfined` tego samego UID
   profil ofiary nie dostarcza opisanej ochrony ptrace. Wczesny
   `PR_SET_DUMPABLE=0` w `main()` występuje za późno, aby sam udowodnić brak
   uprzedniej ingerencji.

Istnieje źródłowo uzasadniony kandydat do osobnego review ochrony startu,
bez setuid i bez file capabilities: zatwierdzony ELF należący do root,
z prawem wykonania, lecz bez prawa odczytu dla docelowego UID przez DAC
(z uwzględnieniem grup, ACL i idmapped mountów). W obu odczytanych jądrach
`fs/exec.c:would_dump()` przy odmowie `inode_permission(..., MAY_READ)`
ustawia `BINPRM_FLAGS_ENFORCE_NONDUMP`; `begin_new_exec()` wybiera wtedy
`fs.suid_dumpable` zamiast `SUID_DUMP_USER`. Przy niezależnie wymuszonym
`fs.suid_dumpable=0` lub 2 i opisanym chronionym starcie z managera root
może to zamknąć konkretną lukę nowego attach przed loaderem. To kandydat
**NOT_PROVEN**, wymagający kontroli dokładnego ELF, credentials, braku
wcześniejszego tracera, niezmienności DAC/sysctl i późniejszych przejść,
a także wpływu braku read na mapowania, loader, samoweryfikację i zamknięcie
zależności; execute-only nie oznacza automatycznego oczyszczenia środowiska
loadera. Nie wdrażamy tego wariantu i nie rozwiązuje on granicy kodu Python
w pamięci ani dowodu pełnej hybrydy.

To źródłowo rozpoznana luka gwarancji tej architektury. Nie oznacza, że
na domyślnej konfiguracji każdej VM powiedzie się `PTRACE_ATTACH` z dowolnego
procesu równorzędnego. Odczytany `security/yama/yama_lsm.c` inicjalizuje
`ptrace_scope` jako `YAMA_SCOPE_RELATIONAL` (1) i sprawdza relację przodka
lub wyjątek ptracer. Yama `ptrace_scope`, ancestry, wyjątki ptracer, DAC,
namespace, dodatkowy LSM i seccomp napastnika mogą osobno odmówić operacji.
Nie zatwierdzono tu nowej podstawy zaufania opartej na globalnym Yama lub
ograniczeniu wszystkich procesów atakującego. Każda próba musi najpierw
wykazać uprawnienie DAC/Yama w osobnym procesie kontrolnym. Odmowy Yama nie
wolno raportować jako odmowy AppArmor; w tej sesji test jądrowy **NOT_RUN**.

`may_change_ptraced_domain()` w obu odczytanych jądrach ma dodatkowy fast
path dla `unconfined(tracerl)`. Przejście profilu nie stanowi dowodu
odłączenia istniejącego tracera ani usunięcia jego wcześniejszych zmian.
Późniejszy odczyt `TracerPid=0` i późniejszy dumpable=0 również takiego
retrospektywnego dowodu nie dają. Obecny tracer może wpływać na stan
wykonania, a kontrola dostępu przy nowym attach nie jest kontrolą każdego
późniejszego żądania wcześniej ustanowionej relacji.

`mm/process_vm_access.c` uzyskuje mm przez `mm_access(...,
PTRACE_MODE_ATTACH_REALCREDS)`; `kernel/pid.c:__pidfd_fget()` sprawdza
`ptrace_may_access(..., PTRACE_MODE_ATTACH_REALCREDS)`. Te operacje korzystają
z tej samej granicy dumpability/LSM i wymagają osobnych przypadków testowych.
Nie zakłada się, że `pidfd_getfd` jest bezpieczny tylko dlatego, że klient ma
socket-bound PIDFD. Po prawidłowo ustanowionym dumpable=0, bez wcześniejszego
tracera i uprawnień nadrzędnych, jądrowa kontrola ptrace stanowi ochronę przed
nowym dostępem; jej rzeczywiste uruchomienie pozostaje **NOT_PROVEN**.

## Brak, tryb i konfiguracja polityki

| Sytuacja | Rzeczywiste znaczenie źródeł i wymagana decyzja przed próbą |
| --- | --- |
| AppArmor wyłączony | Systemd v255 `mac_apparmor_use()` odczytuje `enabled`; `AppArmorProfile=` jest wtedy ignorowane. **BLOCKED** przez zaufany preflight poza badaną aplikacją. Sam wpis jednostki nie zapewnia tego wyniku. |
| Profil niezaładowany | Przy aktywnym AppArmor nieignorowany błąd `aa_change_onexec` kończy start jako `EXIT_APPARMOR_PROFILE`; nie wolno użyć prefiksu `-`. Natywna próba nadal wymaga ustalenia przyczyny i dziennika. |
| Complain | Nazwa profilu może zostać przyjęta, a niewymienione operacje mogą być tylko audytowane. Jawne deny mogą nadal odmawiać; pojedyncza odmowa nie dowodzi enforce. Cała próba **BLOCKED** przed startem. |
| Nieudana zmiana profilu | Odmowa bez fallbacku. Nie dopuszcza się restartu bez polityki lub ostrzeżenia z dalszym działaniem. |
| Jednostka z `-profile`, prefiksem `+` albo zmienionym ExecStart | Kontrakt AppArmor może być pominięty; **BLOCKED**. Sprawdza się efektywną konfigurację, w tym drop-iny, i zaufanie do managera, nie tylko plik na dysku. |
| Brak obsługi składni/features | **BLOCKED**, nawet gdy pakiet parsera jest zainstalowany. Nie stosuje się cichego odrzucania nieobsługiwanej części profilu. |

`AssertSecurity=apparmor` może wymagać aktywnego mechanizmu i dać błąd
jednostki; `ConditionSecurity` może tylko pominąć start. Żadna z tych
właściwości nie poświadcza dokładnych reguł i trybu załadowanego profilu.
`aa-status`, `/proc/PID/attr/current`, `SO_PEERSEC` i nazwa profilu są
obserwacjami pomocniczymi. Nie tworzą niezależnego dowodu zamknięcia procesu,
a deklaracja samej aplikacji nie zastępuje zaufanego preflight.

## Fork, exec, sygnały i przekazanie deskryptorów

**ACCEPT mechanizmu dziedziczenia, REJECT utożsamienia go z brakiem delegacji.**
AppArmor przenosi etykietę przez przygotowanie credentials i kontekst zadania
(`apparmor_cred_prepare`, `apparmor_task_alloc`); zwykły fork/clone może
stworzyć dziecko pod tym samym profilem. Dziecko ma inny PID i może dziedziczyć
listener/połączenie. Profil nie daje samodzielnie wymaganego zakazu fork.
`ix` dziedziczy etykietę przy exec; `px`/`cx` przechodzą do profilu, a ich
warianty i fallbacki trzeba przejrzeć indywidualnie. Żaden dozwolony późniejszy
exec nie jest zgodny z wymogiem ciągłości handlera bez osobnego dowodu.

Sygnały również mają kierunkową mediację i zależą od peer label. Jawna zgoda
na SIGTERM/SIGKILL od systemowego managera działającego jako unconfined może
dać także innemu unconfined procesowi dopuszczonemu przez DAC możliwość DoS.
Nie wolno nazywać takiej reguły uwierzytelnieniem nadawcy systemd. Śmierć,
restart lub zmiana okresu usługi prowadzą do odmowy klienta.

W Ubuntu diff istnieje fine-grained AF_UNIX (`af_unix.c`, odpowiednie hooki
w `lsm.c`). `apparmor_file_receive()` sprawdza uprawnienia odbiorcy do
przyjmowanego pliku/socketu; `aa_unix_file_perm()` ponownie sprawdza część
uprawnień. Nie jest to niezależny zakaz wysyłania SCM_RIGHTS przez proces
posiadający dozwolony kanał IPC. Odbiorca unconfined i dozwolony kanał nie
mogą zostać uznane za niezdolne do przyjęcia FD tylko na podstawie etykiety
nadawcy lub oryginalnego `SO_PEERCRED`.

Faza B musiałaby zabraniać w całości `sendmsg`, `sendmmsg`, `recvmsg` i
`recvmmsg`, a także fork/vfork/clone/clone3 i exec. Klasyczny seccomp nie
inspektuje bezpiecznie `cmsghdr` pod wskaźnikiem; kontrola samego argumentu
pointer nie dowodzi braku SCM_RIGHTS. Przed fazą B nie może istnieć listener,
połączenie, współdzielona tablica FD lub obcy posiadacz deskryptora.
Zakaz późniejszego wysyłania nie odbiera praw wcześniej delegowanych.
Wymagana jest też odmowa zewnętrznego `pidfd_getfd`; filtr tylko w verifierze
nie filtruje syscalli wykonywanych przez napastnika. Pełny dowód delegacji:
**NOT_PROVEN**.

## io_uring oraz mapowania

**REJECT twierdzenia, że profil AppArmor 4 sam zakazuje wszystkich io_uring.**
Dokumentacja `apparmor.d.pod` mówi o ograniczonej mediacji `sqpoll` i
`override_creds`. Potwierdza to `lsm.c:profile_uring()` oraz hooki
`uring_sqpoll` i `uring_override_creds` w obu odtworzonych jądrach Ubuntu.
Brak pozwolenia na te dwie operacje nie oznacza odmowy zwykłego utworzenia
ringu i wszystkich jego operacji. Operacje plikowe mogą przechodzić przez
odpowiednie kontrole LSM, ale nie daje to równoważności z filtrem seccomp
wywołań zlecającego procesu ani zatwierdzenia każdego opcode.

Faza B powinna odmawiać `io_uring_setup`, `io_uring_enter`,
`io_uring_register`. Nie wolno wnioskować z tego, że odcięto wcześniejsze
zlecenia, zarejestrowane zasoby, mapowania albo SQPOLL. Zaufana faza A musi
wykluczać ich utworzenie i odziedziczenie. AppArmor nie cofa wcześniejszego
współdzielonego mapowania; odmowa późniejszego open/mmap nie odbiera
zdalnemu procesowi prawa zapisu do już współdzielonej pamięci. Tych dowodów
nie uzyskano. Próby wcześniejszych mapowań i ringów są **NOT_RUN**.

## Granica Pythona i wynik przeglądu

AppArmor ogranicza dostęp do plików i zasobów. Seccomp ogranicza syscall'e.
Żaden nie rozpoznaje wykonania nowych obiektów kodu CPython, `eval`, `exec`,
interpreterowego dispatch lub wcześniej obecnych bajtów w pamięci.
Usunięcie importów z dysku, brak PROT_EXEC i zatwierdzona nazwa interpretera
nie dowodzą braku niezatwierdzonego kodu Python w pamięci. Jeżeli właśnie to
jest wymaganą gwarancją, pozostaje ona **REJECT**. Nie zastępuje się jej
hookiem audytu ani deklaracją aplikacji.

| Wniosek platformowy | Decyzja | Dowód i granica |
| --- | --- | --- |
| Przejście AppArmor przed pierwszą instrukcją loadera/aplikacji | ACCEPT | Mechanizm `bprm`/ELF opisany wyżej; konkretny start VM NOT_PROVEN. |
| Ochrona loadera przed dowolnym unconfined procesem tego samego UID samym profilem ofiary | REJECT | Obejście mediacji w `profile_tracee_perm`/`profile_tracer_perm`, zwykły exec przywracający dumpable. DAC/Yama mogą osobno blokować atak. |
| Odebranie istniejącemu tracerowi wcześniejszego wpływu przez przejście profilu | REJECT | `may_change_ptraced_domain` pomija unconfined tracera; brak retrospektywnego dowodu. |
| Domyślne fail-closed przez sam AppArmorProfile= | REJECT | Wyłączony AppArmor jest no-op; complain nie daje całej polityki enforce. |
| Zgodność składni i mechanizmów z rodziną Noble GA | ACCEPT | Parser 4.0.1, źródła Ubuntu -31/-149 i systemd v255; nie jest to kwalifikacja efektywnego hosta. |
| Rzeczywiste odmowy jądra tej polityki | NOT_PROVEN | Nie uruchomiono zatwierdzonej VM ani profilu. |
| Brak delegacji FD i wcześniejszych ringów/mapowań | NOT_PROVEN | Zależny od całego zaufanego startu i dokładnego filtra. |
| Zakaz niezatwierdzonego kodu Python w pamięci | REJECT | Brak takiej mediacji w AppArmor/seccomp. |

Minimalny dalszy krok to osobna decyzja architektoniczna: albo zmiana
wymagań tożsamości po niezależnym review, albo podłoże chroniące proces także
przed loaderem wobec napastnika dopuszczonego przez DAC/Yama i zamknięty mały
komponent o przejrzanym protokole. Dedykowany UID, globalna polityka ptrace,
nowy natywny launcher lub inna domena izolacji są zmianą założeń, a nie
zatwierdzonym w tej pracy obejściem. Natywna kwalifikacja wymaga osobno
zatwierdzonej jednorazowej VM z dokładnym kernelem/pakietami i kontrolami
przypisującymi każdą odmowę do faktycznej warstwy ochrony.
