# Stage 9 — macierz adversarialna zamknięcia verifiera TEST_ONLY

Baza: `6b0eb5b92d3f083c1ea2a996b7213be6cf348bc2`, po PR #3251.
Wynik architektoniczny: **REJECT pełnej ochrony wariantu A na tym etapie**.
Stage 9: **IN_PROGRESS / PRODUCTION_NO_GO**.

[Projekt i diagram granicy](../architecture/stage9_test_only_verifier_seal.md)
opisują proponowany launcher. [Model](../../scripts/diagnostics/stage9_verifier_seal_model.py)
nie jest launcherem: nie wykonuje syscalli, nie uruchamia interpretera ani
usługi i nie tworzy socketów. [Testy](../../tests/scripts/test_stage9_verifier_seal_model.py)
sprawdzają wyłącznie dane cBPF i syntetyczną kolejność. Faktyczne wyniki pytest
i kontroli jakości podaje [raport](../audits/stage9_verifier_seal_test_only_2026-11-05.md).

## Znaczenie wyników

- **ACCEPT / REJECT**: oczekiwana decyzja konkretnej granicy w danym scenariuszu.
- **MODELED**: decyzję obliczył model w Pythonie; jądro nie wykonało operacji.
- **Odmowa jądra**: wymaga zarejestrowania dokładnego syscalla, argumentów,
  wyniku/errno lub sygnału, aktualnego ABI i niezależnego dowodu pochodzenia
  filtra. Sam EPERM nie przypisuje odmowy badanemu filtrowi.
- **NOT_RUN**: brak wykonania właściwej próby. Nigdy nie oznacza PASS ani SKIP
  pozytywnego scenariusza bezpieczeństwa.
- **PASS / FAIL / SKIP** w pytest: wynik testu narzędziowego; PASS testu modelu
  pozostaje MODELED, a nie natywnym ACCEPT.

W całej poniższej macierzy kolumna odmowy jądra to **NOT_RUN**: nie ma osobno
zatwierdzonej jednorazowej VM. Nie ustanawiano seccomp, dumpable, NNP,
AppArmor ani nowej polityki hosta. Wcześniejsze regresje jednego UID z lokalnymi
AF_UNIX/pidfd nie dowodzą żadnej z tych odmów.

## Macierz

| Przypadek | Oczekiwanie dla projektu A | Obecny wynik modelowy | Rzeczywista odmowa jądra / kwalifikacja |
| --- | --- | --- | --- |
| Pełna kolejność zatwierdzonego startu → dumpable → importy → FD → NNP → filtr TSYNC → socket/bind/listen | ACCEPT wyłącznie przy dowodzie wszystkich warunków | ACCEPT / MODELED dla zadeklarowanych założeń, `native=NOT_RUN` | NOT_RUN |
| `execve` innego programu lub tego samego Pythona | REJECT | REJECT / MODELED, nr 59 | NOT_RUN |
| `execveat`, również `AT_EMPTY_PATH` | REJECT | REJECT / MODELED, nr 322 niezależnie od argumentów | NOT_RUN |
| `fork`, `vfork` | REJECT | REJECT / MODELED, nr 57 i 58 | NOT_RUN |
| `clone`, również CLONE_THREAD/CLONE_FILES; `clone3` | REJECT | REJECT / MODELED, nr 56 i 435 bez wyjątku dla wątków | NOT_RUN |
| SCM_RIGHTS: eksport listenera przez `sendmsg` | REJECT | REJECT / MODELED dla całego syscalla, nr 46; bez czytania `msg_control` | NOT_RUN |
| SCM_RIGHTS: eksport połączonego socketu przez `sendmsg` / `sendmmsg` | REJECT | REJECT / MODELED dla całych syscalli, nr 46 i 307 | NOT_RUN |
| Odbiór obcego FD przez `recvmsg` / `recvmmsg` | REJECT | REJECT / MODELED dla całych syscalli, nr 47 i 299 | NOT_RUN |
| Odziedziczony listener lub połączenie; socket activation | REJECT przed własnym socketem | REJECT / MODELED, jeśli scenariusz deklaruje odziedziczony listener albo pozostający FD | NOT_RUN; brak wykrywania rzeczywistych odziedziczonych zasobów |
| Niezatwierdzone FD 0, 1, 2 lub inne po zakończeniu przygotowania | REJECT | REJECT / MODELED, jeśli `remaining_fds` nie jest puste | NOT_RUN; brak rzeczywistego `close_range` i inwentaryzacji |
| `ptrace`, `pidfd_getfd`, `process_vm_*` wywołane przez zamknięty proces | REJECT | REJECT / MODELED przez domyślną odmowę | NOT_RUN |
| `ptrace` / `pidfd_getfd` wywołane przez zewnętrzny proces tego samego UID po zamknięciu | REJECT przez uprawnienia celu i dumpable=0 | Brak modelu zewnętrznej kontroli ptrace; filtr celu nie filtruje syscalli napastnika | NOT_RUN |
| Napastnik tego samego UID pomiędzy exec a pierwszym `dumpable=0`; wcześniejszy tracer | REJECT bezpieczeństwa projektu bez dowodu ochrony startu | Deklaracja niezerowego `tracer_pid` daje REJECT / MODELED; późniejsze zero nie dowodzi braku modyfikacji | NOT_RUN; nierozwiązana luka architektoniczna |
| Późniejsze `PR_SET_DUMPABLE=1`, zmiana credentials lub namespace | REJECT | REJECT / MODELED dla `prctl`, zmian UID/GID i namespace | NOT_RUN |
| `io_uring_setup`, `io_uring_enter`, `io_uring_register` | REJECT | REJECT / MODELED, nr 425–427 | NOT_RUN |
| Odziedziczony ring, mapowania lub wcześniejsza praca SQPOLL | REJECT startu | Brak modelu pracy asynchronicznej/mapowań; samo puste `remaining_fds` nie wystarcza | NOT_RUN |
| Plikowy import przez `open`, `openat`, `openat2` po filtrze | REJECT | REJECT / MODELED dla całych syscalli | NOT_RUN |
| Niezatwierdzona closure interpretera/aplikacji przed filtrem | REJECT | REJECT / MODELED tylko gdy założenie `approved_closure` jest fałszywe; model nie mierzy plików | NOT_RUN |
| Niezatwierdzony kod Python z danych już w pamięci, `eval`/`exec`/obiekt kodu | REJECT wymaganej gwarancji wariantu A | **Nie jest rozróżniany przez cBPF**; brak syscalla odpowiadającego wykonaniu bytecode | NOT_RUN; nierozwiązany audyt aplikacji |
| `mmap`/`mprotect` z PROT_EXEC, plikowy mmap | REJECT | REJECT / MODELED; kontrola górnych słów argumentów i anonimowego MAP_PRIVATE | NOT_RUN; nie chroni to przed bytecode Python w pamięci |
| Nowy socket innej rodziny/typu lub bez CLOEXEC | REJECT | REJECT / MODELED; wyłącznie AF_UNIX/STREAM/CLOEXEC i protokół 0 | NOT_RUN |
| Obce ABI, i386 `socketcall`, x32, nieznany numer syscalla | REJECT | KILL lub DENY jako **wartość modelu**, bez sygnału i errno jądra | NOT_RUN |
| Podmiana procesu/MainPID/UID/GID/start-time/okresu podczas wymiany | REJECT | REJECT / MODELED przy zmianie syntetycznego snapshotu | NOT_RUN |
| Podmiana ścieżki lub inode socketu | REJECT | REJECT / MODELED dla zmiany syntetycznego inode/braku ochrony | NOT_RUN; model nie wykonuje lookup ani kontroli DAC/ACL |
| Restart albo śmierć procesu podczas wymiany | REJECT; po pierwszym send `OUTCOME_UNKNOWN`, bez retry | REJECT / MODELED przy zmianie okresu/życia pidfd; brak modelu wysyłki i rollbacku | NOT_RUN |
| Exec lub delegacja FD przy identycznych peer credentials/pidfd/okresie | REJECT pełnej tożsamości bez dowodu ciągłości | Snapshoty pozostają ACCEPT / MODELED; **to pokazuje niewystarczalność obserwacji**, nie skuteczność ochrony | NOT_RUN |
| Błąd dumpable, NNP, instalacji filtra/TSYNC albo dowolnego wcześniejszego kroku | REJECT przed listen, zakończenie bez fallbacku | REJECT / MODELED zatrzaskiwany do końca sekwencji; późniejsze listen również odrzucone | NOT_RUN; nie wywołano prawdziwego listenera |
| Samo `Seccomp: 2` zamiast potwierdzonego kroku filtra | REJECT | REJECT / MODELED; taki krok nie istnieje w sekwencji | NOT_RUN |

## Warunki przyszłego wykonania

Natywne próby wymagają osobnej zgody operatora na konkretną jednorazową VM
Ubuntu Server 24.04. Istniejący guard VM, manifest próby, chronione zasoby,
odrębne UID, offline izolacja i zapis wykonania pozostają wymagane. Ten model
nie tworzy zgody, manifestu próby ani raportu natywnego PASS. Zmiana kernela,
ABI, CPython, glibc, zależności lub filtra wymaga ponownej kwalifikacji.

Przed wykonaniem trzeba uzgodnić ochronę startu, zatwierdzić pełną closure,
przejrzeć dokładne bajty launchera/filtra i odtworzyć kontrolę negatywną bez
badanego ograniczenia. Jeżeli kontrola również daje EPERM, nie można przypisać
odmowy filtrowi. Kontrola nie może ujawniać rzeczywistych danych lub FD usługi.
Potrzebne są także pozytywny scenariusz ograniczonej wymiany oraz niezależny
zapis, że nieudane ustanowienie ochrony poprzedzało każdy socket/bind/listen.

AppArmor wymaga osobnej macierzy rzeczywistych odmów od startu, aktywnej
polityki enforce, przejść profilu i zachowania przekazanych FD. Nie instalowano
ani nie ładowano profilu. Żaden wynik modelu, CI lub etykieta LSM nie zatwierdza
integracji produkcyjnego klienta, AccountGenesis ani przygotowania PostgreSQL.
