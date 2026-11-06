# S9-ID-02 — wykonalność E07/E08 dla obecnego verifiera Python

6 listopada 2026 r.; sprawdzony `origin/main` i baza:
`e488f41a2fd92e6c8895c95929937b94b1fa5020` (merge #3255).
Ocena techniczna: **NOT_PROVEN**. Etap 1: **PENDING_OWNER_ACCEPT**.
K4-v2: **nieaktywny, niezatwierdzony**; Stage 9:
**IN_PROGRESS / PRODUCTION_NO_GO**, `STAGE9_PRODUCTION_READY=NO`.
Merge #3255 nie stanowi zgody na implementację. To ograniczony przegląd
[kontraktu 1.0.0](../architecture/stage9_verifier_identity_acceptance_contract_v1.md),
bez nowego ADR, klienta, launchera, profilu ochrony lub kwalifikacji hosta.

Przeczytano `AGENTS.md`. Codebase Memory nie występuje w katalogu narzędzi ani
jako lokalne CLI; brak grafu i `check_index_coverage`. Poniższe ustalenia dotyczą
bezpośrednio sprawdzonych źródeł. **F** oznacza fakt z kodu; **H** — hipotezę
wymagającą śladu dokładnego runtime/VM. Testy jednostkowe nie zamieniają H w F.

## Rzeczywista kolejność i zasoby

Odnośniki wskazują plik i początek badanego symbolu; numery linii dotyczą bazy.
Granica seal poniżej jest **proponowana**: obecny kod jej nie wykonuje.

| Faza | Fakty: symbole, procesy, pliki i FD | Relacja do przyszłego seal / brak dowodu |
| --- | --- | --- |
| systemd → ExecStartPre | F: [unit:9–37](../../deployment/systemd/cryptohunter-freshness-verifier.service#L9), identyczny `UNIT_TEMPLATE` [runtime:50](../../deployment/freshness_runtime.py#L50): `Type=simple`, systemowy `/usr/bin/python3 -I -s -B`, potem osobny interpreter runtime; różne polecenia oznaczają osobne procesy. NNP, MDWE, puste capabilities, AF_UNIX, native ABI. | E07 musi objąć oba procesy, zmianę credentials, exec i loader. H: dokładne syscalle managera, FD 0/1/2, journald, namespace i dziedziczenie wymagają obserwacji systemd 255. `active` nie dowodzi seal. |
| ExecStartPre → bootstrap | F: `main`, [bootstrap:110](../../deployment/freshness_runtime_bootstrap.py#L110), przez `runpy` sprawdza runtime oraz aplikację; aplikacja ponownie sprawdza runtime. **3 wywołania OpenSSL** na udanej ścieżce. | Wszystkie przed seal; zakończenie ExecStartPre nie uszczelnia późniejszego procesu ExecStart. |
| Podpisy i inwentarze | F: `verify_ed25519`, [bootstrap:31](../../deployment/freshness_runtime_bootstrap.py#L31): 3 memfd na wywołanie, `pass_fds`, `/usr/bin/openssl pkeyutl`, DEVNULL, stałe env/cwd, timeout 15 s, zamknięcie memfd w `finally`. `verify_runtime` [runtime:594](../../deployment/freshness_runtime.py#L594) i `verify_installed_application` [application:163](../../deployment/freshness_application.py#L163) czytają podpisy/klucze/binding z `/etc/cryptohunter`, manifesty i całe drzewa `/opt/cryptohunter/freshness-runtime/releases/...` oraz `/usr/lib/cryptohunter`. `_read` zamyka FD w `finally`. | `pass_fds` to dziedziczenie, nie SCM_RIGHTS. H: fork/vfork/clone/posix_spawn zależy od CPython i argumentów; nie przypisano konkretnego syscalla. H: loader/OpenSSL może otwierać cache, konfigurację i providerów. Zamknąć dzieci i robocze memfd przed seal. |
| ExecStart → interpreter → importy | F: `main` i `_verify_application`, [entrypoint:37](../../deployment/freshness_verifier_entrypoint.py#L37): ponowne sprawdzenie aplikacji i runtime, **2 następne wywołania OpenSSL**, hash/dev/inode `/proc/self/exe`, sztuczny pakiet `bot_core` bez jego `__init__`, import usługi. Importy stdlib występują wcześniej; prefix pycache `/dev/null` ustawiany dopiero w `main`. | `-B` nie wyłącza odczytu pyc, `-I/-s` nie oznacza `-S`. Wcześniejszy interpreter/stdlib/site musi być zaufany zewnętrznie. Wszystkie podpisy i wymagane importy przed seal. |
| Importy i inicjalizacja verifiera | F: [service:13–61](../../bot_core/freshness_verifier_service.py#L13) → `linux_service_ipc` → `production_signer_ipc`, `linux_service_postgresql`, `freshness_semantic_verifier` → `postgresql_freshness_authority`; cryptography/Ed25519 i psycopg. `main` importuje pwd/grp, rozwiązuje dwa UID i GID, sprawdza euid/odrębność UID. Konstruktor verifiera nie rozgrzewa kryptografii. | H: NSS (`nsswitch.conf`, passwd/group, biblioteki/IPC), lazy importy i inicjalizacja crypto/libpq mogą otwierać dalsze pliki lub tworzyć zasoby. Kod aplikacji nie tworzy jawnie wątków ani dzieci w obsłudze; nie dowodzi to zachowania rozszerzeń. |
| PG i diagnostyka przed bind | F: `service.main:50–85`: testowe połączenie PG zamykane przez context, potem opcjonalna polityka `linux-qualification.json`. `load_installed_qualification_policy` [ipc:294](../../bot_core/linux_service_ipc.py#L294) czyta ją przez dirfd/ACL; brak pliku wyłącza diagnostykę. `capture_process_identity` [ipc:67](../../bot_core/linux_service_ipc.py#L67) czyta proc/stat/status/cgroup/exe; qualifier ponownie obserwuje self. | PG i diagnostyka otwierają FD przed proponowanym seal; wrażliwe zasoby wymagają ochrony już wtedy. To nie są retained połączenia do późniejszych żądań. |
| socket → bind/listen | F: `serve_local_unix_socket`, [semantic:653](../../bot_core/freshness_semantic_verifier.py#L653): stat/lstat rodzica, unlink starego socketu tego samego UID, socket AF_UNIX, **bind:700**, chown/chmod 0660, **listen(8):706**. | Nie ma wywołania seal, sprawdzenia wszystkich FD/dzieci/ringów/mapowań/tracera ani wszystkich wątków. Seal musi poprzedzać także bind. Metadata socketu i potrzebne syscalle po seal wymagają osobnego projektu. |
| accept → żądanie → PG → odpowiedź | F: `serve_local_unix_socket:708–774`: nowy FD, SO_PEERCRED; zły UID odrzucony przed recv, root może trafić do odrębnego qualifiera. Sekwencyjny handler czyta `!I` i 2..4 MiB, trzy obiekty base64url/JCS, weryfikuje i odsyła wynik. `verify_exact_candidate` [semantic:362](../../bot_core/freshness_semantic_verifier.py#L362) na poprawnej ścieżce wykonuje **3 weryfikacje Ed25519 i 4 nowe sesje PG**: dwa resolve, predecessor, prepare. | Po seal nadal potrzebne: accept/getsockopt/recv/send/close, alokacje, kryptografia oraz socket/connect i I/O PG. Qualifier [ipc:245](../../bot_core/linux_service_ipc.py#L245) dodatkowo otwiera proc/fdinfo/exe oraz PG przez callback [service:67](../../bot_core/freshness_verifier_service.py#L67). Nie wolno deklarować, że połączenie PG przed bind usuwa te potrzeby. |

W sprawdzanej ścieżce aplikacji nie ma jawnego io_uring ani SCM_RIGHTS.
To obserwacja źródeł, nie egzekwowany zakaz: FFI/RCE i kod zależności nie są
ograniczone samym brakiem wywołania w Pythonie. Nie wykonano śladu kernela.

## Niezgodności E07/E08 i najmniejsze proponowane zmiany

Zmiany poniżej nie są zaimplementowane ani zatwierdzone; wymagają osobnego
zakresu po decyzjach właściciela. Nie aktualizuje się przy tej ocenie pinów.

| Wymaganie | Dokładny dowód / etap i przyczyna | Najmniejsza propozycja naprawy |
| --- | --- | --- |
| E07: ochrona od narodzin | Unit i `UNIT_TEMPLATE` powyżej nie wiążą startu z aktywacją Yama ani dowodem braku wcześniejszego tracera. Późne dumpable nie zabezpiecza loadera ani wcześniejszych importów. | Uzgodnić D03: zaufany boot i rodzic, bramka przed oboma procesami, ochrona także restartów, inventory dziedziczenia i READ_FSCREDS; dopiero później N10/N11 na VM. |
| E08: exec/fork po seal | `serve_local_unix_socket:699–706` binduje bez seal; NNP/native ABI/MDWE nie są zakazem exec/fork/clone. `verify_ed25519:59–83` wymaga subprocess w startującym procesie. | Dokończyć pięć podpisów/startowych procesów przed seal; egzekwować zakaz execve/execveat/fork/vfork/clone/clone3 przed bind. Brak potrzeby subprocess w przejrzanym handlerze przemawia za wykonalnością tej zmiany. |
| E08: delegacja FD, ringi, ABI | Ten sam start nie zamyka obcych FD, wcześniejszych listenerów/ringów/mapowań/dzieci i nie zakłada polityki sendmsg/sendmmsg/recvmsg/recvmmsg/pidfd_getfd/io_uring. Native ABI z unitu nie jest dowodem treści filtra ani TSYNC. | Wczesny, zamknięty inventory dziedziczenia; odrzucenie skażonego startu; końcowe usunięcie roboczych zasobów; egzekwowanie dla wszystkich wątków i ABI. Samo close FD ringu nie dowodzi usunięcia mapowań/zleceń. |
| E08: zamknięte importy/biblioteki | `_external_inventory` [builder:243](../../deployment/build_freshness_runtime.py#L243) i `verify_runtime:594` sprawdzają pliki/stdlib/DT_NEEDED; nie dowodzą wszystkich dlopen/NSS/providerów. `_openssl` [bootstrap:20](../../deployment/freshness_runtime_bootstrap.py#L20) sprawdza DAC ścieżki, nie pełne zatwierdzone closure programu OpenSSL. `_PROBE` buildera działa w innym procesie. | Prześledzić pierwszy i dalsze poprawne/ujemne requesty dokładnego runtime. Wszystkie potrzebne importy/biblioteki załadować przed seal; zaufane mapowania pozostają, a brak późniejszego lazy loading trzeba wykazać dla korpusu i profilu. Sam zatwierdzony inventory nie zamyka importów. |
| E08: pamięć/wątki | `entrypoint.main` i `service.main` nie kontrolują wspólnych mapowań, liczby wątków ani odziedziczonych zasobów. MDWE nie blokuje wykonywania bytecode Pythona. | Zaufany start bez współdzielonej pamięci/obcych kanałów, dumpable=0, polityka bez możliwości cofnięcia ochrony; pojedynczy zweryfikowany wątek lub TSYNC całej grupy. T7 pozostaje jawnym ryzykiem D02. |
| E08: protokół i PG pod ograniczeniami | `PostgreSQLVerifierConnection.connect` [semantic:153](../../bot_core/freshness_semantic_verifier.py#L153) potrzebne po listen; qualifier ponawia open proc. Filtr zamykający wszystkie open/connect zrywa te ścieżki. | Osobno ustalić profil rzeczywistej usługi; zachować kontrolowany dostęp do PG i potrzebnych odczytów albo przejrzeć zmianę lifecycle. Nie wybierać stałej sesji bez analizy izolacji/awarii i semantyki transakcji. Pierwszy prototyp tożsamości nadal bez PG. |
| N15/N17: nadmiarowe bajty żądania | `serve_local_unix_socket:728–752` nie wymaga EOF po pierwszej ramce przed prepare; qualifier `serve_connection:250` wymaga EOF. Jedna wykonana operacja nie dowodzi odrzucania nadmiarowego requestu. | Zamrozić wektory dodatkowych ramek/bajtów i wynik przed wprowadzeniem polityki; wykazać zgodność całego protokołu, nie tylko dodatni challenge. |
| E08: błąd ochrony musi zatrzymać proces | `serve_local_unix_socket:716–724,760–773` połyka błędy diagnostyki, OSError i ogólny Exception, zwraca UNAVAILABLE i kontynuuje. To właściwe rozdzielenie błędów requestu, ale połknęłoby przyszły wyjątek integralności umieszczony w tym try. | Seal musi zakończyć się przed listenerem; odrębna, fatalna ścieżka utraty integralności, poza obsługą zwykłych odmów protokołu. Przyszły test ujemny musi wykazać brak endpointu/odczytu po błędzie ochrony. |
| Sprzątanie i ograniczenie czasu | `connect:155–168`: błąd SELECT po udanym connect nie zamyka jawnie connection. `serve_local_unix_socket:774–776`: sendall poza try; unlink nie jest w finally. `_receive_exact:779` i PG nie mają wspólnego deadline. | Deterministyczny close w błędzie connect, cleanup pathname w finally, rozróżnienie fatal/request; ograniczyć handler i PG bez obietnicy rollbacku po niepewnym COMMIT. Są to propozycje do osobnej zmiany. |

[Sonda #3253](../../scripts/diagnostics/stage9_hybrid_seccomp_probe.c#L78)
dopuszcza po seal tylko exit/exit_group i write na FD 1; odrzuca nawet
socket/bind/listen/accept/mmap. Nie uruchamia Pythona ani PG i nie tworzy
listenera. [Model #3252](../../scripts/diagnostics/stage9_verifier_seal_model.py#L103)
nie instaluje filtra; nie dopuszcza open/connect/getsockopt i potrzebnych
zmian metadanych socketu. Żaden nie dowodzi zgodności rzeczywistej usługi.
Bez wykonania dokładnego closure pod docelową polityką E08 nie jest dowiedzione.

Python nie wymusza exec ani nowych wątków na każde żądanie w tym kodzie.
Nie wykazano bariery uzasadniającej migrację do Rust. Wymóg wykluczenia dowolnego
RCE/FFI/T7 byłby innym, absolutnym wymaganiem i nie uzyska dowodu przez samą
zmianę języka. Akceptowalny zakres przebudowy startup/closure/PG pozostaje
decyzją właściciela; brak dowodu kosztu nie oznacza, że jest on akceptowalny.

## Yama2: zakres D03

Podstawy mechanizmu: [dokumentacja Yama](https://docs.kernel.org/admin-guide/LSM/Yama.html),
`yama_ptrace_access_check`, `yama_ptrace_traceme`, `yama_dointvec_minmax`
w [upstream v6.8](https://github.com/torvalds/linux/blob/v6.8/security/yama/yama_lsm.c),
`__ptrace_may_access` w [ptrace.c](https://github.com/torvalds/linux/blob/v6.8/kernel/ptrace.c),
`process_vm_rw_core` w [process_vm_access.c](https://github.com/torvalds/linux/blob/v6.8/mm/process_vm_access.c),
`__pidfd_fget` w [pid.c](https://github.com/torvalds/linux/blob/v6.8/kernel/pid.c)
oraz `proc_fd_access_allowed`, `mem_open`, `mem_rw` w
[proc/base.c](https://github.com/torvalds/linux/blob/v6.8/fs/proc/base.c).
To źródła mechanizmu, nie kwalifikacja dowolnego kernela Ubuntu GA/HWE;
powiązanie z pakietami Ubuntu wymaga ponownego sprawdzenia na wybranej VM
(istniejący [przegląd platformy](../architecture/stage9_hybrid_platform_review.md)).

| Przy aktywnym Yama=2, bez CAP_SYS_PTRACE napastnika w userns celu | Granica |
| --- | --- |
| Blokuje nowy ATTACH/SEIZE, process_vm_readv/writev, pidfd_getfd oraz obce otwarcie proc-mem przez kontrole ATTACH. PR_SET_PTRACER nie daje wyjątku trybu 1. | TRACEME sprawdza capability rodzica; zaufany uprzywilejowany rodzic pozostaje TCB. Kontrole wewnątrz wspólnej grupy wątków/mm i wcześniejsze zasoby wymagają osobnego zamknięcia. |
| Nie obejmuje samodzielnie READ_FSCREDS: m.in. proc/root/cwd/fd/exe. | DAC/dumpable/inne LSM nadal działają; samo Yama2 nie odmawia tych dostępów. `/proc/PID/fd` nie kopiuje automatycznie socketu, lecz konkretne pliki/dirfd/mounty mogą ujawniać zasoby. |
| Nie blokuje sygnałów/DoS, SCM_RIGHTS, exec/fork/clone, io_uring ani wykonania bytecode/FFI. | To zadanie E08, inventory i zaakceptowanej granicy T7. Nie cofa wcześniejszego tracera, proc-mem FD ani skutków wcześniejszej ingerencji. |
| Wartość 2 może obniżyć uprawniony administrator. | To nie nieodwracalny tryb 3; wymagane zaufanie root i wykazanie ciągłości polityki całej epoki. |

**Może spełnić komponent ochrony startowej E07**, jeśli wartość 2 działa przed
narodzinami pierwszego ExecStartPre i ExecStart. Standardowy start od wartości 1
nie wyklucza takiego profilu: ustawienie 2 w zaufanym bootcie przed usługą musi
być jawnie zaakceptowanym doprecyzowaniem D03. Yama2 **samo nie spełnia całego E07**.
Dodatkowo potrzebne: zaufany rodzic i env, brak wcześniejszych kanałów i tracera,
brak capabilities napastnika, kontrola dziedziczenia oraz polityka READ_FSCREDS.
Przyszłe wrażliwe zasoby nie mogą być osiągalne przez tę ostatnią drogę przed
dumpable=0; późne prctl nie naprawia wcześniejszego wycieku. Mount namespace
ofiary/ProtectProc ani samo odrębne konto nie dowodzą izolacji wobec T3 unconfined.

Proponowany dowód aktywacji na **osobno zatwierdzonej jednorazowej VM**:

1. Chroniony obraz, dokładny kernel/config/LSM, wszystkie sysctl.d i ich
   pierwszeństwo, brak późniejszego override oraz zaufany boot przed obcymi UID.
2. Bramka po zastosowaniu polityki potwierdza Yama=2 i aktywny LSM; oba polecenia
   usługi, restart i ręczny start zależą od jej sukcesu. Samo `After` porządkuje,
   lecz nie propaguje porażki. `systemd-sysctl` v255 jest
   [Before=sysinit.target](https://github.com/systemd/systemd/blob/v255/units/systemd-sysctl.service.in),
   co samo nie dowodzi skutecznego zapisu.
3. Niezależny kontroler wiąże boot ID, monotoniczny czas sukcesu bramki,
   narodziny procesów/exec/loader, InvocationID, cgroup i łańcuch rodzica;
   start ticks mają skończoną rozdzielczość, więc potrzebny też dowód zależności.
   Audyt zmian polityki obejmuje całą epokę, nie tylko końcowy odczyt sysctl.
4. N10/N11: kontrola dodatnia, synchronizacja faz startu, przypisanie odmów
   O-KERNEL i odmowa startu przy złej/niedostępnej polityce. Seria nietrafionych
   wyścigów i końcowe TracerPid=0 nie dowodzą historii.

gdb/strace attach i TRACEME bez wymaganej capability przestaną działać, także
dla własnych dzieci; część profilerów/crash-handlerów wymaga osobnej oceny.
Yama2 nie jest ogólnym zakazem perf/eBPF ani core dumps; dumpable=0 dodaje osobne
ograniczenia. Nie dodawać CAP_SYS_PTRACE usługom dla wygody diagnostyki.
Niezależny kontroler i tryb recovery wymagają uzgodnienia w D03/D08.
Nie zmieniano sysctl, AppArmor ani uprawnień obecnego hosta.

## Dane i metoda D07 — nadal PENDING

| Wielkość | Dostępny dowód | Brak / metoda uzyskania danych |
| --- | --- | --- |
| p50/p95/p99 | Brak wyników benchmarku rzeczywistej usługi w sprawdzonych testach i workflow runtime. Testy poprawności nie dają tych kwantyli. | Zamrożony korpus poprawnych/ujemnych żądań, rozmiary ramki do 4 MiB, pierwsze i ciepłe żądanie, zadeklarowane obciążenie/kolejka. Mierzyć od wywołania klienta do wyniku z E09.post; osobno admission, IPC/crypto, każde PG/COMMIT i recheck. |
| Cała wymiana 3,0 s | `RuntimeFreshnessVerifierClient.verify_exact_candidate` [client:295](../../bot_core/freshness_runtime_ipc.py#L295), `EXCHANGE_TIMEOUT_SECONDS=3.0` [signer_ipc:24](../../bot_core/production_signer_ipc.py#L24); jeden deadline, jedno połączenie, brak retry, po rozpoczęciu send niepewność → OUTCOME_UNKNOWN. | Deadline sprawdzany przed/po operacjach nie preemptuje CPU/hash/proc/stat ani blokującego PG na serwerze. Nie jest dowodem twardego powrotu w ≤3,0 s. Mierzyć zewnętrznym zegarem także przeciążenie, częściowy frame i zablokowany PG/COMMIT; nie zwiększać progu. |
| Start | Pięć OpenSSL × timeout 15 s na wywołanie, hashowanie drzew, importy, NSS i PG. `RestartSec=5s`, `StartLimit*` dotyczą restartów. | Brak budżetu end-to-end; 75 s nie jest limitem całego startu. Od żądania systemd przez ExecStartPre do seal/bind/listen i pierwszej poprawnej wymiany; rozdzielić cache zimny/ciepły i restarty. `Type=simple` nie mierzy gotowości aplikacji. |
| RSS / CPU | Unit nie definiuje liczbowych MemoryMax/CPUQuota; testy nie mierzą RSS/CPU usługi. Workflow [runtime:65](../../.github/workflows/freshness-runtime.yml#L65) ma 20 min na job, nie budżet verifiera. | Zewnętrzny pomiar cgroup: peak/current memory i CPU usage/user/system dla całej jednostki, także ExecStartPre/OpenSSL; osobno RSS/HWM procesu głównego po starcie i po serii, CPU na request/obciążenie. Zapisać konfigurację vCPU/RAM/PG/dysku, wersje i narzut obserwatora. |
| Istniejące limity diagnostycznego SQL | `observe_service_connection` [postgresql:207](../../bot_core/linux_service_postgresql.py#L207): statement/idle-in-transaction 5000 ms, lock 1000 ms. | Dotyczą tylko odczytowej gałęzi root, po connect; nie ograniczają głównego resolve/prepare ani całej wymiany do 3 s. Nie używać ich jako budżetu verifiera. |
| Liczba powtórzeń | Brak zatwierdzonego n, docelowej precyzji ani akceptowalnej częstości przekroczeń. `test_real_timeout_is_bounded_and_unknown_without_retry` [test:425](../../tests/security/test_freshness_runtime_ipc.py#L425) używa 0,5 s i asercji <2 s, nie mierzy budżetu 3 s rzeczywistej usługi. | Do decyzji: pilotaż, następnie n dobrane do przedziałów ufności i zależności czasowej. Kandydat planu: 10 000 wymian na warstwę/scenariusz daje około 100 obserwacji ogona 1%; to nie gwarancja dokładności p99 ani zatwierdzone minimum. Osobne n dla startu, dobierane po pilotażu. |

Rejestrować wszystkie timeouty/odmowy, maksima, liczbę próbek i przedziały
ufności; nie usuwać niepowodzeń z rozkładu. Dla niezależnych prób i zera
przekroczeń jednostronna granica 95% na częstość wynosi
`1 - 0.05^(1/n)`; **299 prób** pozwala zejść poniżej 1%, ale nie wyznacza
precyzyjnego p99. Autokorelacja wymaga niezależnych serii/bloków i korekty
niepewności. Liczby te są metodą planowania, nie wynikiem ani ACCEPT D07.
Porównać bazę i wariant z ochroną przy tej samej platformie i korpusie;
pomiary narzędziami ptrace nie mogą milcząco osłabiać Yama badanego wariantu.
Bez rzeczywistego środowiska nie ustalono limitów RSS/CPU/startu/kwantyli.

## Werdykt i decyzja następnego kroku

| Wymaganie | Dowód z kodu | Wykonalność | Blokada | Najmniejszy kolejny krok |
| --- | --- | --- | --- | --- |
| E07 | Unit/start nie dowodzi ochrony od narodzin; Yama2 obejmuje ATTACH, nie pełne READ_FSCREDS/dziedziczenie. | **NOT_PROVEN**; komponent YAMA2: FEASIBLE_WITH_CHANGES. | Brak zaakceptowanej polityki i natywnej historii startu. | D03: dokładny profil bootu, inventory, dodatkowa polityka i oracle N10/N11. |
| E08 | Pięć subprocess przed proponowanym seal, brak jawnych subprocess/wątków w handlerze, lecz brak seal i cztery PG sesje/request. | **FEASIBLE_WITH_CHANGES** jako kierunek; obecna aplikacja nie spełnia E08. | Closure/lazy I/O, egzekwowanie wszystkich wątków/ABI, fatalne błędy i zgodność protokołu niezakwalifikowane. | Przejrzeć kolejność startup/closure i plan N11–N13/N17, bez implementacji klienta. |
| D03 | Yama upstream i systemd-sysctl dostarczają mechanizmy, obecny unit nie dowodzi ich historii. | **NOT_PROVEN / PENDING**. | Zgoda właściciela i niezależnego recenzenta, wpływ hostowy/diagnostyka/recovery/READ_FSCREDS. | Rozstrzygnąć zaufany boot przed oboma procesami i odrębną VM. |
| D07 | Znane 3,0 s/4 MiB, testy funkcjonalne; brak benchmarków usługi. | **NOT_PROVEN / PENDING**. | Brak liczb RSS/CPU/startu/kwantyli i zaakceptowanego n. | Uzgodnić metodę, pilotaż i budżety; nie zastępować ich czasem pytest/job CI. |

Łącznie **NOT_PROVEN**, bez podstaw do odrzucenia Pythona ze względu na język.
Zastosowanie istniejącej sondy wprost do usługi jest **REJECT**: uniemożliwia
jej protokół. Osobno autorytet: stały `/run/postgresql` i peer-only
[pg_hba](../../deployment/postgresql/pg_hba.conf)/[pg_ident](../../deployment/postgresql/pg_ident.conf)
nie odróżniają obcego procesu UID verifiera; bez dodatkowej izolacji byłby
**AUTHORITY_REJECT**, niezależnie od przyszłego dowodu tożsamości.

Najbliższy krok to decyzje D02/D03/D04/D07/D08 właściciela i niezależnego
recenzenta na podstawie konkretnych braków powyżej; pełny ACCEPT nadal wymaga
D01–D08. Dopiero osobna zgoda i ACCEPT dopuszczają ograniczony eksperyment
TEST_ONLY na wskazanej jednorazowej VM, najpierw bez PG. Nie uruchomiono go tutaj.

## Walidacja i zachowany zakres

Zmiana dotyczy wyłącznie tego raportu. Bez zmian klienta/verifiera/AccountGenesis,
PG, pinów/podpisów/manifestów i `deployment/stage9_current_status.json`.
Nie restartowano workflow ani nie wykonano testów uprzywilejowanych.
Zestaw istniejących testów kontraktu, IPC/tożsamości, instalacji/runtime/aplikacji,
semantyki i modelu seal: **1049 PASS / 24 SKIP** w lokalnym CPython 3.11.17.
Pominięcia: 21 prób rzeczywistego PG wymagających root, dwa jawne root harnessy
i brak rzeczywistych dodatkowych grup konta. Nie są PASS kwalifikacji natywnej.
Pierwsze uruchomienie ograniczała piaskownica (`EPERM` przy bind AF_UNIX);
powtórzenie z dostępem do własnych socketów było nieuprzywilejowane.
Dokładny hook `ruff-format --all-files`, Ruff, skonfigurowane mypy Linux/Win32
(po 392 pliki), Betterleaks 1.9.0 offline z baseline i pełną redakcją oraz
kontrola odnośników/diff: PASS. To walidacja raportu/regresji, nie benchmark.
Polecenia walidacji i bieżący stan CI są w opisie osobnego Draft PR;
brak merge. Natywne N01–N18 pozostają **NOT_RUN** dla K4-v2.
