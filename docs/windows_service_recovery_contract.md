# Windows service recovery — kontrakt backendu

Decyzja produktu z 2026-11-02: backend ma automatycznie uruchamiać bounded
recovery zarówno po crash bez zgłoszenia `SERVICE_STOPPED`, jak i po
`SERVICE_STOPPED` z niezerowym **`dwWin32ExitCode`**. Zwykły stop z kodem 0
nie kolejkuje nowej akcji recovery. Właściciel produktu zatwierdził to
rozszerzenie podczas analizy PR #3241; wcześniejsze wymagania potwierdzały
wyłącznie kryterium crash i same nie uzasadniały zmiany flagi.

## Źródła i rozstrzygnięcie

| Źródło | Zakres wymagania |
| --- | --- |
| [`platform_readiness.json`](../deployment/platform_readiness.json), `semantic_invariants`, `acceptance.WINDOWS_AUTOMATIC_CRASH_RESTART` | Nadzór crash, graceful shutdown, autostart i osobne kryterium restartu po crash. |
| Scalony [PR #2945](https://github.com/Dudzian/Dudzian/pull/2945), Motivation | Automatyczny restart po crash z ograniczoną polityką failure actions. |
| [`process_topology_and_lifecycle.json`](architecture/cryptohunter_product_architecture/process_topology_and_lifecycle.json), `failure_matrix`, `restart_policy`, `valid_core_lifecycle_pairs` | Crash CoreHost uruchamia nadzorowany restart; strategie czekają na rekoncyliację, kill switch pozostaje trwały. Semantyczne `STOPPED` i `CRASHED` są oddzielne. |
| [`README` architektury](architecture/cryptohunter_product_architecture/README.md), „Status M0.3 — closed” | M0.3 jest kontraktem semantycznym, nie implementacją Windows Service. Nie definiuje translacji natywnych kodów SCM na stany produktu. |
| [`windows_scm_probe.ps1`](../deployment/windows_scm_probe.ps1), `CRASH_RESTART`, `POST_RECOVERY_GRACEFUL_STOP` | Probe wymusza crash przez `Stop-Process -Force`, obserwuje nowy PID bez ręcznego startu oraz zwykły stop. Nie kwalifikuje błędnego `STOPPED`. |
| Decyzja produktu z 2026-11-02 | Recovery obejmuje także `SERVICE_STOPPED/nonzero`; zwykły stop z kodem 0 nie kolejkuje restartu. |

Brak ustawienia flagi w poprzednim kodzie nie stanowił dowodu świadomej polityki
`FALSE` ani gwarancji wartości zastanej na hoście. Dopiero powyższa decyzja
uzasadnia jawne ustawienie i sprawdzanie `TRUE`. Analiza korzystała z wyszukiwania
i bezpośrednich odczytów źródeł; Codebase Memory i kontrola pokrycia jego indeksu
nie były dostępne.

## Semantyka Windows SCM

Macierz zakłada skonfigurowane failure actions oraz już skuteczną wartość flagi.
„Tak” oznacza kwalifikację zdarzenia do kolejki akcji; rzeczywisty restart zależy
od licznika awarii i bieżącej akcji, w tym końcowego `SC_ACTION_NONE`.

| Zakończenie usługi | Skuteczne `FALSE` | Skuteczne `TRUE` — zatwierdzony kontrakt |
| --- | --- | --- |
| Proces kończy się bez zgłoszenia `SERVICE_STOPPED`, także z kodem procesu 0 | Tak | Tak |
| `SERVICE_STOPPED`, `dwWin32ExitCode=0` | Nie | Nie |
| `SERVICE_STOPPED`, `dwWin32ExitCode=1066` (`ERROR_SERVICE_SPECIFIC_ERROR`) | Nie | Tak |
| `SERVICE_STOPPED`, inny niezerowy `dwWin32ExitCode` | Nie | Tak |

Sam niezerowy `dwServiceSpecificExitCode` przy `dwWin32ExitCode=0` nie spełnia
warunku. Bez failure actions flaga nie uruchamia żadnej akcji. Zwykły
`STOPPED/0` nie kolejkuje nowej akcji, lecz nie anuluje restartu już
zakolejkowanego po wcześniejszej awarii. Źródła:
[Microsoft, SERVICE_FAILURE_ACTIONS_FLAG](https://github.com/MicrosoftDocs/sdk-api/blob/656e9656bfaef34dd29cd7ee9e1146aec17e924c/sdk-api-src/content/winsvc/ns-winsvc-service_failure_actions_flag.md#L61-L71),
[ChangeServiceConfig2W, Remarks](https://learn.microsoft.com/en-us/windows/win32/api/winsvc/nf-winsvc-changeserviceconfig2w).

Przypięty `pywin32==312` mapuje nieobsłużony wyjątek `SvcRun()` na
`SERVICE_STOPPED/1066/0x20000001`, a zwykły powrót na `SERVICE_STOPPED/0/0`.
`service_base.py` pozostawia końcowe zgłoszenie natywnemu hostowi; nowa flaga
uzupełnia kontrakt recovery, nie zmieniając własności statusu końcowego.
Źródło: [`PythonService.cpp`, service_main](https://github.com/mhammond/pywin32/blob/2a277cb5552756c2b4d42b524dc36d25e0bb6354/win32/src/PythonService.cpp#L834-L857).

## Konfiguracja, inspekcja i aktywacja

`windows_installer.provision._recovery()` oraz właścicielsko ograniczony
`windows_service_recovery.configure_recovery()` zapisują
`SERVICE_CONFIG_FAILURE_ACTIONS_FLAG=True` i wymagają odczytu dokładnego
Pythonowego `True`. Dla tego poziomu pywin32 używa pojedynczego `bool`,
nie słownika. `False`, `None`, liczby, ciągi i słowniki nie przechodzą weryfikacji.
Źródła przypiętego ABI: [`setter`](https://github.com/mhammond/pywin32/blob/2a277cb5552756c2b4d42b524dc36d25e0bb6354/win32/src/win32service.i#L1637-L1643),
[`getter`](https://github.com/mhammond/pywin32/blob/2a277cb5552756c2b4d42b524dc36d25e0bb6354/win32/src/win32service.i#L1724-L1727).

Sekwencja pozostaje ograniczona do restartów po 1 s / 5 s / 30 s, następnie
`SC_ACTION_NONE`, z resetem licznika po 86400 s, bez command/reboot.
Uchwyty setterów obejmują `SERVICE_START`, którego SCM wymaga przy
`SC_ACTION_RESTART`; samo posiadanie prawa nie uruchamia usługi.

Inspekcja hosta odczytuje flagę z query-only uchwytu i zapisuje ją w polu
`failure_actions_on_non_crash_failures` obserwacji usługi. Backend poza
trybem `clean_base` musi mieć dokładnie `True`; brak, niewłaściwy typ i
niezgodność kończą inspekcję odmową. PostgreSQL, verifier i clean-base
pozostają poza wymaganiem polityki restartu backendu. `--verify-only`
acceptance helpera nie zmienia konfiguracji ani nie żąda praw change/start.

Microsoft dokumentuje skuteczność zmiany flagi **przy następnym starcie
systemu**. Zapis i zgodny readback potwierdzają zapisaną konfigurację,
a nie aktywację w bieżącym boot ani wykonany restart. Restart samej usługi
nie zastępuje dowodu przejścia systemu. Wyniku statycznej inspekcji nie wolno
opisywać jako żywej kwalifikacji recovery. Wdrożenie na istniejącym hoście
wymaga osobno zaplanowanej aktywacji; ta zmiana nie wykonuje rebootu.

## Sterowanie statusem i anulowanie

Wspólna granica `service_base.py` serializuje raportowanie statusu oraz start
endpointu przez `RLock`. Zapamiętuje pełny status dopiero po udanym
`ReportServiceStatus`. `SvcInterrogate` odtwarza ten status przez natywne
`SetServiceStatus`, zachowując controls, checkpoint i wait hint bez uruchamiania
endpointu ani wykonywania kwalifikacji readiness. Samo interrogation nie
zwiększa checkpointu: pywin32 zwiększa go przy high-level raporcie pending,
a pytanie SCM nie dowodzi postępu workera. Przed pierwszym udanym zgłoszeniem
handler nie wymyśla statusu.

Żądanie `STOP_PENDING` utrwala osobny znacznik zatrzymania przed próbą zgłoszenia
do SCM, również gdy to zgłoszenie zawiedzie. Późne raporty readiness nie mogą
cofnąć usługi do `START_PENDING` ani `RUNNING`; interrogation ponawia wtedy
`STOP_PENDING` zamiast odtwarzać stary `RUNNING`. Po zakończeniu lub awarii
`SvcRun` Python nie zgłasza kolejnych statusów. Wyłącznym właścicielem końcowego
`SERVICE_STOPPED` pozostaje natywny host pywin32.

`SvcStop` i delegujący do niego `SvcShutdown` niezależnie podejmują próbę
zgłoszenia `STOP_PENDING`, ustawienia `stop_event`, ustawienia `evidence_stop`
oraz natywnego `SetEvent`. Pierwszy wyjątek zachowuje tożsamość; dodatkowe
błędy są dopisywane wyłącznie przez stałą nazwę operacji i nazwę klasy,
bez kopiowania ich tekstu do notatek lub logów. Operacje anulowania i join
endpointu nie trzymają blokady statusu.

Deterministyczne modele wymuszają kolejność współbieżnych INTERROGATE/STOP
przez zdarzenia, bez opierania poprawności na opóźnieniach. Integracja modeluje
handler kontrolny oddzielnie od workera PostgreSQL: błąd handlera nie jest
sztucznie propagowany przez `WaitForSingleObject`. Worker otrzymuje anulowanie,
opuszcza monitorowanie oryginalnego uchwytu i wykonuje cleanup. Testy zachowują
również błędne zakończenie postmastera z kodem 0 i niezerowym oraz pojedynczy
natywny `STOPPED`. Polityka recovery i jej bramki pozostają bez zmian.

## Walidacja i granice dowodu

Testy setterów obejmują zapis `True` dla początkowych `False` i `True`,
niezgodny lub wadliwy readback, błędy natywnego API, zamknięcie uchwytów,
niezmienione bounded actions i odmowę dla niewłaściwej tożsamości.
Testy inspekcji obejmują odczyt obu wartości flagi, wymóg `True` backendu,
negatywne typy/brak pola, wyjątek przy drugim odczycie SCM i query-only prawa.
Modele rozróżniają crash, `STOPPED/0`, `STOPPED/1066`, inne błędy oraz zapisaną
i aktywną flagę. Testy lifecycle osobno modelują końcowy status natywnego hosta.

Model i readback nie kwalifikują rzeczywistego restartu na Windows. Nie zmieniono
bramek ownership, legal enrollment, Stage 9/10 ani gotowości produkcyjnej.
Nie wykonano produkcyjnego provisioningu, restartów usług, rebootu ani natywnej
kwalifikacji recovery. Żywa próba po aktywacji wymaga oddzielnego, izolowanego
Windows i istniejących ręcznych bramek kwalifikacji; zwykłe PR CI jej nie wykonuje.

## Anulowanie readiness PostgreSQL

`postgresql_service.wait_ready()` zwraca jawny wynik: `True` dla gotowości,
`False` dla zaobserwowanego anulowania. Rzeczywisty upływ deadline nadal rzuca
`PostgreSQLReadinessTimeout`. Ustawione zdarzenie anulowania przed następną
próbą kończy oczekiwanie bez kolejnego probe, także gdy upłynął deadline.

Każde ukończone wywołanie `pg_isready` jest poprzedzone i zakończone odczytem
oryginalnego uchwytu postmastera. Zaobserwowany exit 0/niezerowy lub błąd API
pozostaje awarią również przy równoczesnym anulowaniu; nie przechwytujemy
wyjątków procesu lub subprocess jako normalnego anulowania. Dopiero po
końcowym odczycie uchwytu sprawdzane są anulowanie, deadline i wynik probe.
Sukces probe po deadline nie potwierdza gotowości. Te obserwacje nie dowodzą
fizycznej kolejności równoczesnych zdarzeń procesu i anulowania.

Worker z anulowaniem ustawionym na wejściu nie tworzy potomka. Po utworzeniu
procesu anulowanie pomija `RUNNING` i monitor, ale przechodzi przez istniejące
`pg_ctl stop -m fast`, potwierdzenie zakończenia oryginalnego uchwytu oraz
cleanup uchwytów thread/process/job. Stan anulowania jest ponownie sprawdzany
bezpośrednio przed publikacją readiness. Błędy zatrzymania, potwierdzenia exit
i cleanup nadal propagują awarię z zachowaniem pierwszej przyczyny. Dotyczy to
także możliwego braku pidfile podczas wczesnego rozruchu; nie maskujemy błędu
`pg_ctl` jako udanego anulowania.

**Ograniczenie SCM:** standardowa usługa w `SERVICE_START_PENDING` nie akceptuje
zwykłego STOP. Przypięty host pywin32 zgłasza wtedy `dwControlsAccepted=0`;
tej polityki nie zmieniono. Nowe testy wstrzykują programowe zdarzenie workera,
nie wywołują `SvcStop` w START_PENDING i nie dowodzą osiągalności tej ścieżki
przez natywny SCM w produkcji. Nie dodano nowego mechanizmu dostarczania
anulowania podczas rozruchu. Natywne potwierdzenie wymaga osobnego izolowanego
Windows oraz konkretnego, zatwierdzonego źródła anulowania. Źródła kontraktu:
[SERVICE_STATUS](https://learn.microsoft.com/en-us/windows/win32/api/winsvc/ns-winsvc-service_status)
i [ControlService](https://learn.microsoft.com/en-us/windows/win32/api/winsvc/nf-winsvc-controlservice).
