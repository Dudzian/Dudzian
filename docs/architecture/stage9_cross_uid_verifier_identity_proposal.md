# Stage 9 — propozycja tożsamości verifiera między odrębnymi UID

Status: **PROPOZYCJA DO NIEZALEŻNEJ AKCEPTACJI ARCHITEKTURY**.
Dokument nie zatwierdza rozwiązania, nie zmienia kontraktu klienta i nie jest
dowodem kwalifikacji. PR #3248 pozostaje Draft, `NO MERGE`,
`STAGE9_PRODUCTION_READY=NO`.

## Problem i dotychczasowa granica

`os_freshness_runtime` ma uwierzytelnić rzeczywisty endpoint usługi
`os_freshness_crypto_verifier`, nie jedynie istniejący socket lub proces
tego samego UID. Obecny transport wymaga socket-bound `SO_PEERPIDFD`,
credentials, czasu startu, cgroup i dokładnego executable. Przy różnych
UID odczyt `/proc/PID/exe` zwykle jest zabroniony przez kontrolę ptrace.
Brak tego odczytu nadal oznacza odmowę przed wysłaniem żądania.

Istniejące `freshness_service_identity.py` sprawdza efektywną jednostkę,
ExecStart/ExecStartPre, podpisany runtime oraz bieżący obraz procesu.
`ExpectedPeerProcess` jest zaufanym wejściem, ale samo przekazanie
zewnętrznie zaobserwowanego PID/hashu nie zastępuje bieżącej kontroli obrazu.
Nie należy tworzyć go z niezweryfikowanej odpowiedzi serwera.

Root instalacji, systemd, zatwierdzony bootstrap i superuser PostgreSQL
pozostają dotychczasową granicą zaufania. Propozycja nie dodaje
`CAP_SYS_PTRACE`, uprzywilejowanego pośrednika ani zamiennika
`SO_PEERPIDFD` opartego na późniejszym `pidfd_open(PID)`.

## Wariant do oceny: zatwierdzony start i egzekwowana ciągłość usługi

Najmniejszym wariantem do osobnego review jest powiązanie jądrowej
tożsamości połączenia z wcześniej zatwierdzonym uruchomieniem usługi,
zamiast przyznawania runtime prawa do obserwacji obcego executable.
To **proponowana zmiana podstawy dowodowej**, a nie równoważna zamiana
sprawdzenia hashu na porównanie PID.

1. W istniejącym, osobno autoryzowanym procesie instalacji powstaje
   chroniony przez root kontrakt: dokładne nazwy UID/GID, ścieżka socketu,
   jednostka/cgroup, zamrożone argumenty uruchomienia i zatwierdzony
   podpisany inwentarz interpretera oraz aplikacji. Runtime ma tylko odczyt.
   Dokument nie tworzy teraz takiego pliku ani nie zmienia manifestów.
2. Zwykły runtime odczytuje przez lokalny systemowy D-Bus dokładne
   właściwości systemd używane przez obecny kwalifikator: aktywność,
   MainPID/ExecMainPID, okres uruchomienia, ExecStart/ExecStartPre,
   efektywne ograniczenia i brak dodatkowej konfiguracji. Trzeba osobno
   potwierdzić dostępność tych odczytów bez nowych uprawnień oraz
   uwierzytelnienie właściwego lokalnego system bus. Nie ufa się danym
   o systemd zwróconym przez verifiera.
3. Na faktycznym AF_UNIX runtime uzyskuje `SO_PEERCRED` i
   `SO_PEERPIDFD`, koreluje PID z zatwierdzonym MainPID, kontroluje życie
   uchwytu i wszystkie niezależnie dostępne credentials/start-time/cgroup.
   Odczyty przed i po wymianie muszą dotyczyć tego samego okresu usługi;
   restart, rozbieżność, brak pola lub brak prawa odczytu oznaczają odmowę.
   Chroniona ścieżka i dokładny protokół pozostają wymagane.
4. Osobny mechanizm musi egzekwować ciągłość od zatwierdzonego startu
   do obsługi żądania: brak podmiany obrazu przez `execve`/`execveat`,
   brak niezatwierdzonego potomka obsługującego odziedziczony listener
   oraz brak przekazania deskryptora połączenia obcemu handlerowi.
   Konkretny mechanizm i jego dopuszczalny zakres wymagają review oraz
   negatywnych testów jądrowych; ten dokument go nie implementuje.

Punkt 4 jest warunkiem wariantu, nie opcjonalnym hardeningiem.
Obecne `NoNewPrivileges`, `ProtectSystem` i odczytowe metadane systemd
same nie dowodzą zakazu ponownego exec, fork ani przekazania deskryptora.
Nie należy przypisywać im takiej semantyki.

## Czego obserwacje nie dowodzą

MainPID, start-time i cgroup mogą pozostać identyczne po exec innego
programu. ExecStart D-Bus opisuje start zlecony przez systemd, nie mierzy
każdego późniejszego obrazu procesu. Cmdline jest mutowalne; argv nie jest
pomiarowym dowodem executable. PIDFD wiąże czas życia procesu z socketem,
ale nie poświadcza hashów wykonywanego kodu ani procesu faktycznie
obsługującego odziedziczony/przekazany deskryptor.

Podpisany inwentarz potwierdza zatwierdzone bajty plików, nie sam fakt ich
wykonywania. Nawet zakaz kolejnego exec nie uwierzytelnia dowolnego kodu
interpretowanego przez Python; potrzebna jest osobna zamknięta, zatwierdzona
granica importów i obsługi danych. Nie uznaje się zgodności hashu Pythona
za dowód właściwej aplikacji.

Akceptacja wariantu wymaga jawnej decyzji, czy zaufanie do zatwierdzonego
startu, egzekwowanej ciągłości i zamkniętej aplikacji spełnia model zagrożeń.
Jeżeli wymagany pozostaje niezależny bieżący pomiar executable, a zwykły
runtime nie może go wykonać, wariant jest niewystarczający. Wtedy wynik
pozostaje `BLOCKED`; nie wolno zastąpić brakującego dowodu samym UID,
MainPID, deklaracją serwera ani snapshotem z instalacji.

## Najmniejszy bezpieczny podział prac

1. **Decyzja architektury:** zatwierdzenie albo odrzucenie powyższej
   zmiany podstawy zaufania, wymaganego modelu wykonania i konkretnych
   dowodów egzekwowania ciągłości. Bez zmiany kodu klienta lub jego odmów.
2. **Osobny prototyp i kwalifikacja:** po akceptacji, odtwarzalna próba
   w izolowanej VM z dwoma rzeczywistymi UID, chronioną instalacją,
   rzeczywistym systemd i istniejącym protokołem AF_UNIX. Próba nie jest
   zwykłym CI PR ani instalacją na Dellu. Wymaga osobnej autoryzacji
   działań administracyjnych; ta propozycja jej nie udziela.
3. **Osobna integracja:** dopiero po review dowodów, dokładny diff
   klienta i AccountGenesis, usunięcie starego retry oraz niezależne
   rozstrzygnięcie przypiętych bajtów, inwentarza i podpisu wydania.
   Pozytywny prototyp nie zatwierdza nowych digestów automatycznie.

Warunki odmów przyszłej próby obejmują obcy UID/GID, złą jednostkę/cgroup,
zmianę MainPID/okresu usługi, restart podczas wymiany, exec tego samego
lub innego interpretera, fork z odziedziczonym listenerem, przekazanie FD,
podmianę socketu/ścieżki/kontraktu, zmieniony inwentarz/import oraz brak
D-Bus, procfs albo `SO_PEERPIDFD`. Przypadek, którego obserwator nie potrafi
odróżnić, ma dawać odmowę lub `BLOCKED`, nie deklarację wykrycia.
Po rozpoczęciu wysyłania niejednoznaczność nadal daje `OUTCOME_UNKNOWN`
bez automatycznego retry. Odmowy muszą mieć dowód braku nieautoryzowanego
przygotowania w rzeczywistym PostgreSQL, osobno od testów modelowanych.

## Blokady pozostają aktywne

Nie zmieniają się zamrożone źródła, piny, podpisy, manifesty, status JSON,
coverage, CI ani Stage 10. Nie podłącza się klienta do produkcyjnego
AccountGenesis. Nadal obowiązują:

- `RUNTIME_IPC_APPLICATION_ADAPTER_UNAVAILABLE`;
- `ACTUAL_RUNTIME_CLIENT_EXCHANGE_UNPROVEN`;
- `RUNTIME_UNAUTHORIZED_IDENTITY_DENIAL_UNPROVEN`;
- `DURABLE_CHALLENGE_REPLAY_PROTECTION_UNAVAILABLE`;
- pełna kwalifikacja IPC i integracji produktu: `BLOCKED`;
- `STAGE9_PRODUCTION_READY=NO`, Draft, `NO MERGE`.
