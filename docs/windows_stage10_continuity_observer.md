# Windows Stage 10 — natywny obserwator ciągłości procesu

Obserwator dostarcza odczytowe obserwacje jednej instancji procesu. Jest podstawą
dalszej pracy nad `same_service_instance`; nie kwalifikuje produkcyjnego lifecycle.
`duration_at_least_24h` pozostaje `BLOCKED`, także po odmierzeniu 86 400 sekund.
`WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE` nadal bezwarunkowo zamyka
bramkę dowodową. Nie zmieniono statusu Stage 9 ani Stage 10.

## Architektura i granica zaufania

Moduł
[`windows_stage10_continuity_observer.py`](../deployment/windows_stage10_continuity_observer.py)
ma dwa rozdzielone wejścia:

| Wejście | Źródło procesu | Zakres wyniku |
| --- | --- | --- |
| `observe_backend(*, seconds, stop=None)` | Wyłącznie `CONTRACT.backend_service` i rzeczywisty lokalny SCM; brak parametru PID lub ścieżki użytkownika. | `SCM_CONTRACT`; obserwacja tożsamości, bez kwalifikacji zatwierdzonego produktu. |
| `observe_test_process(child, *, seconds, stop=None)` | Własny proces pomocniczy reprezentowany przez `subprocess.Popen`. | Zawsze `TEST_ONLY`; nie jest usługą CryptoHunter. |

Ścieżka backendu wykorzystuje istniejący
[`windows_stage9_evidence_pipe._PinnedService`](../deployment/windows_stage9_evidence_pipe.py),
bez zmiany jego kontraktu. Adapter otwiera SCM z `SC_MANAGER_CONNECT`, a usługę
z `SERVICE_QUERY_CONFIG | SERVICE_QUERY_STATUS`. Wymaga działającej usługi
`SERVICE_WIN32_OWN_PROCESS`, kanonicznego konta `NT SERVICE\CryptoHunterBackend`,
kanonicznego obrazu z natywnego folderu Program Files oraz zgodnego SID tokenu.
PID pochodzi z `QueryServiceStatusEx`, nigdy z argumentu operatora. Konfiguracja
i tożsamość procesu są sprawdzane ponownie po przypięciu oraz podczas obserwacji.

`seconds` musi być skończoną, dodatnią liczbą nie większą niż 86 400.
Limit żądanego zakresu nie udostępnia kwalifikacji 24h ani nie uruchamia takiej
próby w CI. Testy natywne żądają wyłącznie krótkich zakresów.

Ta zgodność nie ustanawia pochodzenia zatwierdzonego wydania, rzeczywistej
gotowości backendu, legal production enrollment ani autorytetu PostgreSQL.
Dotychczasowe kwalifikatory tych właściwości i ich odmowy nadal obowiązują.
Obserwator nie przyjmuje manifestu, receipt ani zapisanego wyniku jako zastępczej
tożsamości procesu. Nie wystawia proof ID, `results` ani produkcyjnego `PASS`.

## Uchwyty i używane API Win32

| API | Znaczenie w obserwacji |
| --- | --- |
| `OpenSCManagerW`, `OpenServiceW`, `QueryServiceConfigW`, `QueryServiceStatusEx` przez istniejący adapter pywin32 | Wybór i ponowna weryfikacja rzeczywistej instancji usługi z kontraktu. |
| `OpenProcess` | Jeden uchwyt procesu z `PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE` (`0x00101000`), bez dziedziczenia. |
| `GetProcessTimes` | Natywny czas utworzenia procesu przypiętego uchwytem. |
| `QueryFullProcessImageNameW` | Natywna ścieżka obrazu tego samego procesu. |
| `WaitForSingleObject` | Oczekiwanie na zakończenie dokładnie przypiętego procesu; sygnał kończy obserwację odmową. |
| `CloseHandle`, `CloseServiceHandle` | Zwalnianie uchwytów procesu, tokenu, usługi i SCM w każdej ścieżce wyjścia. |
| `QueryInterruptTimePrecise`, `QueryUnbiasedInterruptTimePrecise` | Para natywnych liczników monotonicznych w jednostkach 100 ns. |

Nowy adapter `ctypes.WinDLL` stosuje wzorzec istniejących adapterów Windows:
ładowanie systemowej DLL z ograniczeniem do System32, jawne `argtypes` i `restype`,
`use_last_error=True` oraz kontrola wyników Win32. Brak wymaganej funkcji lub błąd
wywołania oznacza odmowę obserwacji, bez przechodzenia na UTC lub `psutil`.

Przez cały zakres pozostaje otwarty oryginalny uchwyt. Obserwator nie otwiera
zastępczego procesu, gdy SCM wskaże nowy PID lub ten sam numer PID po restarcie.
Zakończony obiekt procesu pozostaje zakończony także wtedy, gdy system ponownie
przydzieli jego PID. Tożsamość zawiera czas utworzenia i ścieżkę obrazu, a nie
tylko numer PID. Początek i koniec pomiaru należą do tej samej sesji z tym samym
uchwytem; porównanie dwóch plików JSON nie jest sesją obserwacji.

## Wyniki, błędy i przerwania

Niezmienny obiekt `Observation` zawiera stan, zakres, przyczynę,
`observed_100ns`, `process_identity` oraz stałe `duration_24h="BLOCKED"`.
Wynik jest diagnostyką i nie może być wejściem przywracającym dawną obserwację.

| Stan | Znaczenie |
| --- | --- |
| `CONTINUOUS` | Oryginalny proces pozostawał dostępny i niezakończony w zakończonym zakresie obserwacji; wyłącznie w podanym zakresie diagnostycznym. |
| `OBSERVATION_LOST` | Po rozpoczęciu utracono zdolność potwierdzania: błąd uchwytu lub SCM, przerwanie, anulowanie, błąd zegara, luka w kontroli albo niepewne suspend/resume. |
| `PROCESS_EXITED` | Oryginalny uchwyt został zasygnalizowany przez zakończenie procesu. Nowa instancja nie kontynuuje tej obserwacji. |
| `IDENTITY_CHANGED` | Odczytana tożsamość procesu lub przypisanie do usługi przestały być zgodne. |
| `NOT_OBSERVED` | Obserwacja nie mogła się rozpocząć, np. brak natywnego Windows, usługi lub zgodnej konfiguracji. |

Anulowanie przez `stop` oraz przerwanie kwalifikatora kończy sesję. Sprzątanie
odbywa się także po błędzie inicjalizacji i wyjątku; błąd zamykania nie zachowuje
pozytywnej diagnozy. Nie ma wznowienia po restarcie interpretera, sumowania kilku
sesji ani zaliczania brakujących próbek. Nieużyty lub stary wynik nie otrzymuje
władzy nad kolejnym wywołaniem.

## Czas monotoniczny i suspend/resume

`QueryInterruptTimePrecise` obejmuje czas uśpienia i hibernacji;
`QueryUnbiasedInterruptTimePrecise` ten czas wyklucza. Oba źródła zwracają
całkowite jednostki 100 ns, niezależne od korekt daty UTC. Odczyt licznika
unbiased jest otoczony dwoma odczytami biased, aby ograniczyć niepewność parowania.
To podstawa pomiaru czasu obserwacji, nie autorytet czasu UTC ani synchronizacji
zewnętrznej.

Obserwator odmawia po cofnięciu licznika, niespójności ich postępu, zbyt szerokim
parowaniu odczytów (ponad 20 ms), zmianie różnicy biased/unbiased ponad 20 ms
lub luce w kontroli ponad 5 s. Dłuższy suspend/resume lub zatrzymanie obserwatora
przerywa zakres. Opóźnienie planisty może spowodować bezpieczną odmowę.

Tolerancja odczytu oznacza, że bardzo krótki suspend albo przejście na granicy
sesji może nie zostać osobno rozstrzygnięte. Nie ma jeszcze niezależnej, kompletnej
obsługi powiadomień o zasilaniu i związania ich z zakresem próby. Dlatego zegar ma
status `PARTIAL`, a `duration_at_least_24h` zawsze pozostaje `BLOCKED`.
Samo osiągnięcie 86 400 sekund ani liczba wykonanych próbek nie zmieniają tego
statusu. Przyszły kwalifikator musi wymagać co najmniej 86 400 sekund jednej
nieprzerwanej i w pełni potwierdzonej sesji; nie może odtwarzać czasu z UTC.

Semantyka źródeł:
[QueryInterruptTimePrecise](https://learn.microsoft.com/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryinterrupttimeprecise),
[QueryUnbiasedInterruptTimePrecise](https://learn.microsoft.com/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryunbiasedinterrupttimeprecise)
oraz
[WaitForSingleObject](https://learn.microsoft.com/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject).
Wymagane są dostępne natywne funkcje systemu Windows; brak symbolu nie jest
podstawą do użycia słabszego zamiennika.

## Testy i granice ich znaczenia

Oddzielny workflow
[`windows-stage10-continuity-observer.yml`](../.github/workflows/windows-stage10-continuity-observer.yml)
korzysta z hostowanych `ubuntu-latest`, `windows-latest` i `macos-latest`.
Nie korzysta z self-hosted ani środowiska produkcyjnego. Nie uruchamia fizycznej
kwalifikacji, instalacji MSI, restartu, 24h soak, enrollment lub operacji na
kluczach i usługach produkcyjnych.

Testy syntetyczne sprawdzają model stanu, ponowne użycie PID, zmianę obrazu i czasu
utworzenia, błędy SCM i Win32, utratę obserwacji, anulowanie, błędy zegara,
suspend/resume, próby użycia starej diagnozy i zwalnianie zasobów. Ich atrapy
nie dowodzą semantyki rzeczywistych uchwytów Windows ani bezpieczeństwa
zainstalowanej usługi.

Krótkie testy natywne na `windows-latest` tworzą i kończą wyłącznie własne procesy
pomocnicze. Weryfikują otwarcie i utrzymanie oryginalnego uchwytu, natywny czas
utworzenia, wykrycie zakończenia, odmowę zastąpienia starej instancji nową,
sprzątanie oraz odmowy błędów API i niedostępnego SCM. Nie zakładają ponownego
przydzielenia konkretnego PID przez system. Deterministyczny przypadek ponownego
użycia PID pozostaje syntetyczny. Każdy wynik helpera ma zakres `TEST_ONLY`.

Raport pokrycia workflow jest raportem testów kodu. Zielony test helpera nie
kwalifikuje rzeczywistego `CryptoHunterBackend`, ciągłości przez 24h, semantyki
wszystkich stanów zasilania ani reboot/autostart. Wyniki natywne można uznać za
wykonane dopiero po zakończeniu joba Windows dla dokładnego HEAD danego PR;
kolejka CI nie oznacza sukcesu. Istniejące regresje Stage 9/10, CodeQL oraz
kontrole jakości i bezpieczeństwa nadal są odrębnymi bramkami.

## Pozostałe blokady i późniejsza próba na zatwierdzonym hoście

Przed pełnym lifecycle potrzebne są kompletne rozstrzyganie suspend/resume,
chroniona tożsamość jednej próby i hosta, zatwierdzony handoff wydania, bieżąca
kwalifikacja produktu i legal production enrollment, autentykowane obserwacje
gotowości backendu i PostgreSQL oraz osobny dowód reboot i pochodzenia autostartu.
Obserwacja zgodnego procesu SCM nie zastępuje żadnego z tych autorytetów.

Późniejsza próba wymaga najpierw przeglądu implementacji brakujących autorytetów
i osobnej autoryzacji fizycznego wykonania zgodnie z
[procedurą Stage 10](windows_stage10_proof_validation.md) oraz
[read-only preflight](windows_stage10_readonly_preflight.md). Na zatwierdzonym
hoście należy potwierdzić aktualny produkt i runtime istniejącym kwalifikatorem,
rozpocząć nową sesję ze świeżym uchwytem SCM, zebrać autentykowane obserwacje
przed i po i utrzymać kompletny zakres co najmniej 86 400 sekund. Każdy restart
procesu, luka, utrata uchwytu albo nierozstrzygnięte przejście zasilania kończy
próbę odmową i wymaga nowej sesji od zera. Dowody reboot/autostart pozostają
oddzielną próbą i osobnym autorytetem.

Niniejszy moduł nie udostępnia produkcyjnej procedury wydania pozytywnego
receipt. `require_native_proof_authority()` nadal odmawia bezwarunkowo.

`WINDOWS_24H_LIFECYCLE_QUALIFIED=NO`

`WINDOWS_REBOOT_RECOVERY_QUALIFIED=NO`

`STAGE10_PRODUCTION_READY=NO`
