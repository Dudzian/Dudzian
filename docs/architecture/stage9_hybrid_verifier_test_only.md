# Stage 9 — hybryda AppArmor od exec i późny seccomp, TEST_ONLY

5 listopada 2026 r.; baza `main` po PR #3252:
`5cb848d656c2b7d9d8b06e654be7911f3c9fac0f`.
**REJECT pełnej wymaganej ochrony.** Stage 9 pozostaje
**IN_PROGRESS / PRODUCTION_NO_GO**, `STAGE9_PRODUCTION_READY=NO`.

Hybryda ma wykonalne składniki, ale nie ustanawia wymaganej ciągłości kodu
verifiera. [Przegląd źródeł platformy](stage9_hybrid_platform_review.md)
rozstrzyga istotny brak: w zbadanym Ubuntu GA profil ofiary nie odmawia ptrace
ze strony nieograniczonego procesu wyłącznie przez własne `deny ptrace`.
Zwykły exec może ponadto przywrócić dumpable przed pierwszą instrukcją
aplikacji. Dodatkowa odmowa Yama nie jest dowodem działania AppArmor.
Druga niezależna granica pozostaje niespełniona: ani LSM, ani seccomp nie
rozpoznają niezatwierdzonych obiektów kodu wykonywanych przez Python w pamięci.

## Dwie fazy proponowanego projektu

**A — od startu.** Zaufany systemowy manager root, dokładny efektywny unit
bez wyjątków `+`, `!` i pomijania błędów, chroniony argv/cwd/środowisko oraz
zatwierdzone niezmienne bajty kodu są warunkami koniecznymi. Profil ma zostać
przypisany przez `AppArmorProfile=` bez `-` albo jednoznaczne attachment
zatwierdzonego ELF. W obu wariantach etykieta docelowa obowiązuje podczas
ładowania obrazu, przed loaderem i aplikacją; `aa_change_onexec` nie obejmuje
wstecznie kodu managera przed exec. Sama kolejność zmiany UID w systemd
nie dowodzi luki: trzeba uwzględnić reset dumpable i blokady exec. Luka
własnej ochrony AppArmor wobec unconfined tracera wymaga osobnego rozwiązania.

Przyszły profil aplikacji wymaga zamkniętej listy dokładnych plików ELF,
loadera, bibliotek, modułów i zasobów; ścieżki rodziców muszą być chronione
przez DAC przed UID runtime i verifiera. Brak ogólnego `/usr/** r`, bibliotek
z zapisywalnego venv, cwd, home, pluginów, `.pth`, środowiska `LD_*`/`PYTHON*`
i niezweryfikowanego bytecode. Prawa `m` i exec muszą wynikać z przejrzanego
inwentarza. Brak takiego inwentarza oznacza NOT_PROVEN, nie rozszerzenie reguł.
Zakaz ptrace obu kierunków trzeba sprawdzić także wobec unconfined napastnika;
polityka sygnałów i AF_UNIX wymaga review obu stron. Fork dziedziczy profil,
ale dziedziczenie nie jest zakazem potomków, współdzielenia pamięci lub FD.

**B — po zatwierdzonej inicjalizacji, przed listenerem.** Mały zaufany natywny
launcher musi sprawdzić `PR_SET_DUMPABLE=0`/`PR_GET_DUMPABLE`, ustanowić
`NO_NEW_PRIVS`, zamknąć niezatwierdzone FD łącznie z 0–2 i wykazać brak
wcześniejszych dzieci, współdzielonej pamięci, ringów, zleceń i obcych kopii
deskryptorów. Inicjalizacja nie może wcześniej tworzyć listenera. Dopiero po
pełnym zamknięciu importów i zasobów wolno założyć dokładny filtr cBPF
z domyślną odmową dla zatwierdzonego ABI i TSYNC. Każdy niezerowy wynik
instalacji, również dodatni TID przy TSYNC, kończy proces. Potem dopiero
socket/bind/listen; brak socket activation i adoptowania zastanego FD.

Filtr musi zabronić późniejszego exec, wszystkich niezatwierdzonych
fork/vfork/clone/clone3 i wątków, zmiany dumpable/credentials/namespaces,
profilu oraz niepotrzebnego `prctl`/`seccomp`. Zakaz całych `sendmsg`,
`sendmmsg`, `recvmsg`, `recvmmsg` omija problem wskaźnika do ancillary data;
klasyczny seccomp nie czyta `msg_control`. Trzeba odmówić wszystkich trzech
syscalli io_uring i wykluczyć wcześniejsze mapowania/SQPOLL. Profil AppArmor
z badanego kernela nie zastępuje tego ogólnego zakazu.

Wykonanie bytecode nie wymaga nowego exec, pliku ani PROT_EXEC. Usunięcie
`eval`, hook audytu lub ograniczenie importów nie tworzy nieodwracalnej
granicy wobec przejętego interpretera. To wymaganie ma **REJECT** dla
rozpatrywanego modelu, a nie tylko status oczekiwania na kolejne testy.

## Ograniczony prototyp i jego wartość diagnostyczna

[`stage9_hybrid_seccomp_probe.c`](../../scripts/diagnostics/stage9_hybrid_seccomp_probe.c)
jest małym statycznym programem x86-64 LP64. Nie osadza Pythona, nie otwiera
listenera i nie implementuje aplikacji verifiera. Ustanawia rzeczywiste
dumpable=0, NNP i 17-instrukcyjny filtr TSYNC. Pozostawia wyłącznie exit,
exit_group i write na dokładnym 64-bitowym argumencie FD=1; reszta odmawia.
Filtr weryfikuje architekturę i bit x32. Jest celowo dużo węższy niż profil
potrzebny do AF_UNIX lub CPython i nie może zostać użyty jako taki profil.

[Testy](../../tests/security/test_stage9_hybrid_seccomp_probe.py) kompilują
warianty z różnymi errno i zmienioną wyłącznie akcją ochrony x32, dzięki czemu
przypisują obserwowaną odmowę własnemu filtrowi, zamiast zaliczać zastane
EPERM/SIGSYS. Sprawdzają 27 wejść syscalli, odmowę nieprawidłowych flag filtra
oraz nieznanego trybu. Argumenty są celowo nieprawidłowe: to dowód odmowy
wejścia, nie wykonanie rzeczywistego SCM_RIGHTS, iniekcji lub pozytywnej
komunikacji. Statyczna glibc i stan sprzed `main` nie zostały zakwalifikowane
jako bezpieczny start; sonda nie inwentaryzuje FD ani mapowań. Nie jest
bezpiecznym launcherem. Nie używać z produkcyjnymi danymi.

[Profil](../../deployment/test_only/stage9_hybrid/stage9-hybrid-test-only.apparmor)
jest zamkniętym kandydatem **wyłącznie dla tej sondy**: dokładny statyczny
ELF `mr`, `/dev/null rw`, bez innych plików, exec, bibliotek, Python,
capabilities i komunikacji sieciowej/AF_UNIX. Przyjmuje tylko SIGTERM/SIGKILL
od `unconfined`, żeby modelować zakończenie przez managera. To pozwala także
napastnikowi spowodować utratę dostępności; etykieta nie identyfikuje systemd.
Odmowa ptrace nie usuwa opisanej wyżej granicy unconfined w GA.
Nie dodano reguły io_uring udającej brak ogólnych operacji ringu.

[Projekt jednostki](../../deployment/test_only/stage9_hybrid/stage9-hybrid-test-only.service)
ma osobne nieutworzone konto TEST_ONLY, pusty zestaw capabilities i brak
sekcji Install. **Zawsze blokuje wykonanie** przez
`ConditionPathIsDirectory=/dev/null`. `ConditionSecurity=apparmor` jest
dodatkową kontrolą obecności, nie sprawdzeniem enforce. Pominięta przez
systemd jednostka nie jest zaliczeniem próby; oznacza BLOCKED w raporcie.
Nie ma gotowego preflight sprawdzającego aktywną politykę ani kontrolowanej
procedury jej ładowania. Usunięcie blokady wymaga osobnego review jednej VM.
Nie należy uruchamiać ani instalować tych materiałów na bieżącym hoście.

## Tożsamość klienta i istniejące źródła

Przyszły klient hybrydowy musiałby powiązać faktyczny AF_UNIX z SO_PEERCRED
i socket-bound SO_PEERPIDFD, życiem procesu, chronionym inode, boot ID,
start-time, cgroup, MainPID/ExecMainPID i okresem systemd przed/po wymianie.
To wymaganie projektu, nie zaimplementowany ciągły odczyt systemd/boot ID
przez obecnego klienta. Obecny kontrakt akceptacji pozostaje bez zmian.
Odczyt samej nazwy profilu lub SO_PEERSEC niczego tu nie zastępuje.
PIDFD nie atestuje treści filtra ani późniejszego posiadacza przekazanego FD.

Bezpośrednio sprawdzone zależności i dokładne symbole opisuje
[macierz](../runbooks/stage9_hybrid_adversarial_matrix.md#powiązanie-z-istniejącym-verifierem-i-decyzje).
`freshness_runtime_ipc._check_server` nadal odmawia przy brakującym hashu
executable; nie dodano alternatywnej ścieżki akceptacji. Późna niejednoznaczność
pozostaje OUTCOME_UNKNOWN bez retry. Bootstrap korzysta z subprocess OpenSSL,
a produkcyjne `service.main` łączy PostgreSQL przed listenerem; nie zostały
uruchomione przez sondę. Nie zmieniono ich kontraktu, jednostki ani instalatora.

## Decyzja według kryteriów

| Kryterium | Wynik | Dowód i granica |
| --- | --- | --- |
| Bezpieczeństwo startu i ciągłości procesu | **REJECT** | Źródła Ubuntu GA: zwykły exec może przywrócić dumpable, a profil celu nie mediuje ptrace unconfined napastnika. Późny seal nie cofa modyfikacji. |
| Odporność na napastnika tego samego UID | **REJECT** | Samo AppArmor+late seccomp nie daje wymaganej gwarancji; zależność od Yama lub dodatkowej polityki nie jest zatwierdzonym dowodem tej hybrydy. Nie deklaruje się udanego ataku na domyślny obraz. |
| Zamknięcie aplikacji Python i zależności | **REJECT** | Brak zatwierdzonego closure i brak jądrowej granicy dla niezatwierdzonego kodu interpretowanego w pamięci. |
| Rzeczywiste egzekwowanie pełnych zakazów | **NOT_PROVEN** | Natywna sonda potwierdza tylko swój filtr na Debian/kernel 6.18.57; pełne A+B na Noble nie wykonane. |
| Odporność na przekazanie deskryptorów | **NOT_PROVEN** | Odmówiono wejść syscalli sondy; rzeczywiste SCM_RIGHTS, odziedziczenie i io_uring z mapowaniami nie były próbą tego artefaktu. |
| Zgodność z Ubuntu 24.04/systemd 255 | **NOT_PROVEN** | Przegląd konkretnych źródeł i parser Noble nie zastępują uruchomienia dokładnego kernela, unitu i zamkniętej aplikacji. HWE wymagają osobnego sprawdzenia. |
| Wpływ tego PR na istniejący FreshnessVerifier | **ACCEPT** | Wyłącznie nowe materiały TEST_ONLY; żadnej zmiany klienta, produkcyjnych plików, danych ani kontraktu akceptacji. |
| Zakres przyszłych zmian instalatora | **NOT_PROVEN** | Potrzebne zatwierdzone profile/runtime, preflight enforce i cech, cykl aktualizacji/wycofania oraz nowy kontrakt startu. Nie implementowano ich. |

ACCEPT izolacji bieżącej zmiany nie oznacza ACCEPT architektury.
Pełny plan, próby nieprzeprowadzone i kryteria odmowy są w
[macierzy adversarialnej](../runbooks/stage9_hybrid_adversarial_matrix.md).

## Alternatywa i minimalny dalszy krok

Najpierw rozstrzygnąć niezależnie wymaganie odporności na kod Python w pamięci
oraz podstawę ochrony startu wobec unconfined napastnika. Dla niezmienionych
wymagań potrzebna jest istotna zmiana: mały natywny verifier z domkniętym kodem
i przejrzanym parserem, dodatkowa niezależna granica izolacji napastnika oraz
nowe zatwierdzenie kontraktu klienta. Natywny język sam nie dowodzi odporności
na błędy pamięci. Dedykowany UID lub globalna polityka Yama także zmieniają
założenia i wymagają review; nie są ukrytą poprawką tej propozycji.

Przegląd źródeł wskazuje także węższego kandydata dla samego startu:
root-owned execute-only ELF bez prawa odczytu przez UID celu może przez
`would_dump()` / `BINPRM_FLAGS_ENFORCE_NONDUMP` zachować `fs.suid_dumpable`
0 lub 2 podczas exec, bez setuid/capabilities. To **NOT_PROVEN**, wymaga
osobnego review DAC/ACL/mountów, sysctl, loadera i samoweryfikacji.
Nie wdrożono go w prototypie; nie zamyka granicy kodu Python w pamięci.

Po tej decyzji minimalną próbą jest osobno zatwierdzona jednorazowa VM
z dokładnym kernelem, testem przed pierwszą instrukcją launchera i kontrolą
DAC/Yama/AppArmor. Nie ma podstaw do uruchamiania produkcji, podłączenia
AccountGenesis ani scalania tego Draft PR.
