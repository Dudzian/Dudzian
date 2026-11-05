# Stage 9 — klient runtime Freshness przez lokalny AF_UNIX

Baza: `main` po merge PR #3247,
`b387b7a20813c1d5157551320d3fc7f6e1396b32`. Zakres obejmuje wąski adapter
transportu `bot_core/freshness_runtime_ipc.py` do istniejącej operacji
`FreshnessSemanticVerifier`. Nie dodaje operacji serwera ani kwalifikacji
produkcyjnej. Obowiązuje `STAGE9_PRODUCTION_READY=NO`.

## Interfejs i granica zaufania

`RuntimeFreshnessVerifierClient(expected: ExpectedPeerProcess)` otrzymuje
istniejący model oczekiwanego procesu z `deployment/linux_physical_ipc.py`.
`expected` musi pochodzić z niezależnie zatwierdzonej obserwacji właściwej
usługi. Nie wolno tworzyć go przez skopiowanie tożsamości dowolnego procesu
znalezionego przy sockecie. Zaufany kod wywołujący odpowiada za związanie
obserwacji z efektywnym systemd, zatwierdzonym wydaniem i podpisaną aplikacją.
Adapter sam nie ustanawia tych dowodów.

Współrzędne pozostają stałe: Linux, strumieniowy AF_UNIX i
`/run/cryptohunter/freshness/verifier.sock`. Klient wymaga rzeczywistego
odrębnego konta `os_freshness_runtime`; serwer ma należeć do
`os_freshness_crypto_verifier`, z grupą `freshness_verifier_ipc`. Nie ma
parametru ścieżki użytkownika, transportu TCP, wyboru roli bazy, hasła ani
mechanizmu zmiany UID.

Verifier jest autorytetem kryptograficznej weryfikacji i trwałego
przygotowania dokładnego kandydata. Jego PostgreSQL peer mapping prowadzi
do roli `freshness_crypto_verifier`. Runtime ma odrębną rolę
`freshness_runtime` do późniejszego `compare_and_advance`. Sam transport
nie łączy się z PostgreSQL, nie wykonuje CAS i nie przygotowuje kandydata
lokalnie. Główny serwer kontroluje UID runtime przed odczytem żądania
i przed wywołaniem weryfikatora.

## Przypięcie socketu i procesu

Adapter używa istniejącego `_pinned_verifier_socket()` z
`deployment/linux_physical_ipc.py`: przechodzi po chronionych katalogach
przez deskryptory z `O_NOFOLLOW`, przypina rzeczywisty socket oraz ponownie
sprawdza jego inode, właściciela, grupę, tryb, metadane i ACL. Połączenie
odbywa się przez przypięty katalog w `/proc/self/fd`, a nie przez ponownie
rozwiązywaną dowolną ścieżkę.

Istniejący `PinnedReadOnlyPeer` z `bot_core/linux_service_ipc.py` wiąże
`SO_PEERCRED` i `SO_PEERPIDFD` danego połączenia z żywym procesem. Nazwa klasy
odzwierciedla jej pierwotnego konsumenta diagnostycznego; sam mechanizm
sprawdza tożsamość partnera, nie wybiera operacji aplikacji. Adapter nie
wywołuje `READONLY_QUALIFY`.

Klient dodatkowo wymaga zgodności PID, UID/GID, czasu startu i SHA-256
executable z `expected` oraz dokładnego cgroup
`/system.slice/cryptohunter-freshness-verifier.service`. Digest programu
nie może być `None`. Brak dostępu do executable obcego UID, niedostępny
socket-bound PIDFD, zakończenie procesu albo zmiana obserwacji powodują
odmowę. Nie podnosi się uprawnień, aby uzyskać brakującą obserwację.
Tożsamość i przypięta ścieżka są kontrolowane przed wysłaniem żądania
oraz po odpowiedzi. Ponowny odczyt `/proc/PID/cmdline` odrzuca również
zmianę argumentów po `exec` tego samego interpretera.

Przy odrębnych produkcyjnych UID runtime zwykle nie ma prawa ptrace do
odczytu `/proc/PID/exe` verifiera. W tej konfiguracji odmowa przed wysłaniem
żądania jest oczekiwanym wynikiem tego adaptera, nie dowodem działającej
integracji runtime. Pozytywny harness z jednym UID nie rozwiązuje tej
granicy. Zatwierdzenie sposobu powiązania właściwej usługi z nieuprzywilejowanym
runtime wymaga osobnej decyzji architektonicznej; ten PR nie dodaje capabilities,
uprawnień ptrace ani pośrednika uprzywilejowanego.

Hash interpretera Python sam nie uwierzytelnia importowanych modułów.
Zgodność z `expected` i cgroup jest warunkiem transportu; niezależne
zatwierdzenie źródeł i skutecznej jednostki pozostaje odpowiedzialnością
zaufanego wywołującego. Root, superuser PostgreSQL i uprzednio zatwierdzony
bootstrap pozostają poza gwarancją tego adaptera.

## Dokładny protokół i niejednoznaczny wynik

Klient wysyła jedną ramkę istniejącego protokołu: czterobajtową długość
sieciową `!I`, następnie kanoniczny JSON JCS z dokładnie trzema polami
`proposer_authentication`, `authoritative_document`, `finalization_receipt`.
Wartości są kanonicznym base64url bez paddingu. Obowiązuje limit istniejącego
serwera 4 MiB, dokładna liczba ramek i jeden deadline obejmujący wymianę.
JSON używa istniejących `canonical_json_bytes()` i `parse_canonical_json()`;
powtórzone pola, niekanoniczny zapis, liczby niedopuszczone przez kontrakt
i dodatkowe dane nie są akceptowane.

Odpowiedzi aplikacji mają zamknięte kategorie `PREPARED`, `INVALID`,
`UNAVAILABLE`, `OUTCOME_UNKNOWN`. `PREPARED` wymaga dokładnego zestawu pól,
identyfikatora przygotowania i kanonicznego bindingu o istniejącym schemacie.
Poprawny własny hash przygotowania nie wystarcza: klient niezależnie
porównuje binding z trzema niezmiennymi obiektami wysłanymi w tym żądaniu.
Odtwarza istniejące preimages `DOCUMENT_DIGEST_DOMAIN`, `PROPOSER_DOMAIN`
i `DOCUMENT_AUTHENTICATION_DOMAIN`, dokładny hash kanonicznego receipt
(łącznie z jego podpisem) oraz `ORIGINAL_DECISION_DOMAIN`. Porównuje także
wynikające z żądania selektory i tożsamości. Preimages uwierzytelnienia
proposera i dokumentu pomijają pole podpisu zgodnie z protokołem verifiera;
transport nie zastępuje jego weryfikacji podpisów i retained credentials.
Odpowiedź dla innego poprawnego kandydata, nawet z poprawnym własnym
`preparation_id`, daje `OUTCOME_UNKNOWN` po wysłaniu żądania, bez retry.
Adapter nie interpretuje błędu transportu jako negatywnego dowodu
autentyczności kandydata.

Przed rozpoczęciem wysyłania odmowa tożsamości lub niedostępność transportu
oznacza brak wysłanego żądania. Od momentu rozpoczęcia wysyłania brak
jednoznacznej, poprawnej odpowiedzi musi zachować możliwość wykonania
operacji przez autorytet. Obejmuje to timeout, ucięcie, błędną długość lub
JSON, nieznaną kategorię, dodatkową ramkę, nieprawidłowy binding oraz zmianę
tożsamości lub ścieżki po wysłaniu. Klient nie ponawia automatycznie ani
`OUTCOME_UNKNOWN`, ani żadnego takiego niejednoznacznego wyniku.

## Zakres testów i ograniczenia dowodowe

`tests/security/test_freshness_runtime_ipc.py` obejmuje testy adaptera
z niezależnym procesem testowym i rzeczywistymi lokalnymi socketami AF_UNIX,
kernel credentials oraz socket-bound PIDFD. Próby są nieuprzywilejowane.
Fixture modeluje produkcyjne nazwy kont i chronioną ścieżkę, których
utworzenie wymagałoby administracyjnej instalacji; nie jest to dowód
rzeczywistej izolacji dwóch produkcyjnych UID ani uruchomienia właściwej
jednostki systemd. Testy jawnie rozdzielają ten model od rzeczywistych
operacji jądra.

Model instalacji obejmuje także brak dodatkowych grup usługi. Proces
testowy na hosted Ubuntu może dziedziczyć grupy konta runnera, które
produkcyjny guard prawidłowo odrzuca przed wysłaniem żądania. Fixture
podaje guardowi kopię obserwacji z modelowaną grupą usługi, pozostawiając
oryginalne `peer.process`, SO_PEERCRED, SO_PEERPIDFD i `peer.recheck()`
bez zmian. Ponowny odczyt nadal sprawdza rzeczywiste grupy procesu i ich
stabilność. Nie jest to dowód natywnej izolacji grup produkcyjnej usługi.

Odrębny test bez modelu grup potwierdza rzeczywistą odmowę przed send,
gdy konto testowe ma grupy dodatkowe. Na koncie bez takich grup zgłasza
jawne `BLOCKED`/SKIP. Reprodukcja Ubuntu 24.04 z nieuprzywilejowanym
UID i jedną syntetyczną grupą dodatkową pozwala sprawdzić obie ścieżki
bez tworzenia kont, zmiany uprawnień hosta ani osłabienia guardu.

Modele resolvera i writer sprawdzają dojście do właściwej granicy oraz brak
wywołania przygotowania po odmowie autoryzacji lub weryfikacji. Nie są
rzeczywistym PostgreSQL i nie ustanawiają dowodu ACL, peer mapping ani
niezmienności stanu produkcyjnej bazy. Testy ramkowania i odpowiedzi nie
kwalifikują kryptograficznej poprawności syntetycznego autorytetu.
Na nieobsługiwanym systemie klient odmawia. Brak wymaganej kontroli PIDFD
oznacza odmowę lub jawny `BLOCKED`, nigdy pozorny pozytywny dowód transportu.

`tests/security/test_freshness_runtime_ipc_identity.py` zawiera odrębne
modele NSS, rzeczywistych/efektywnych/zapisanych UID/GID, grup i odmów
przed otwarciem transportu. Oba zestawy można odtworzyć bez root:

```bash
python -m pytest -q tests/security/test_freshness_runtime_ipc.py tests/security/test_freshness_runtime_ipc_identity.py --cov=bot_core.freshness_runtime_ipc --cov-branch
python -m ruff check bot_core/freshness_runtime_ipc.py tests/security/test_freshness_runtime_ipc.py tests/security/test_freshness_runtime_ipc_identity.py
python -m pre_commit run ruff-format --all-files
python -m mypy bot_core/freshness_runtime_ipc.py
python -m mypy --platform win32 bot_core/freshness_runtime_ipc.py
```

Potrzebne są zależności testowe repozytorium, `pytest-cov` i Linux
obsługujący `SO_PEERPIDFD`; niedostępny syscall pozostawia jawny `BLOCKED`.
Skonfigurowany mypy całego repozytorium należy uruchomić osobno dla obu
platform. Nowy moduł sprawdza powyższe jawne polecenie; ten PR nie zmienia
inventory ani bramek CI.

Dokładne wyniki wykonania i wszystkie pominięcia należą do raportu PR
dla jego końcowego HEAD. Ten dokument nie przypisuje testom wyniku przed
ich uruchomieniem. Lokalne testy nie stanowią natywnego dowodu właściwej
usługi runtime ani pełnej kwalifikacji IPC produktu.

## Pozostawiona integracja i najmniejszy dalszy krok

Istniejący `ProductionLocalFreshnessAuthority` wykonuje transport AF_UNIX,
ale nie sprawdza tożsamości serwera i automatycznie ponawia przygotowanie
po `PREPARATION_OUTCOME_UNKNOWN`. Nie spełnia kontraktu tego adaptera.
Nowy moduł jest wąskim transportem do istniejącego autorytetu, a nie drugim
wysokopoziomowym autorytetem AccountGenesis.

Ten PR nie podłącza adaptera przez zmianę przypiętych bajtów
`account_genesis_freshness_authority.py`. Osobny audyt po PR #3247 nadal
wykazuje drift jego szóstego pinu; pięć pinów minimalnej usługi pozostaje
zamrożonych. Nie ma automatycznego rehash ani zatwierdzenia nowych bajtów.

Najmniejszy bezpieczny podział dalszych prac zaczyna się od osobnego
zatwierdzenia modelu uwierzytelnienia serwera przez odrębny UID runtime,
z jawnym rozwiązaniem ograniczenia procfs i testem rzeczywistej izolacji.
Do tej decyzji nie rozszerza się uprawnień ani gwarancji adaptera.
Osobny [projekt do niezależnej akceptacji architektury](stage9_cross_uid_verifier_identity_proposal.md)
opisuje warunki takiej decyzji; nie zatwierdza nowej podstawy zaufania.

Kolejna osobna propozycja do autoryzacji obejmuje przegląd dokładnego
diffu AccountGenesis: zastąpienie jego starego transportu wywołaniem
zweryfikowanego klienta, doprowadzenie niezależnie zatwierdzonego `expected`
do rzeczywistego właściciela procesu runtime oraz usunięcie automatycznej
ponownej próby nieznanego wyniku przygotowania. Po niezależnym zatwierdzeniu
tych bajtów należy osobno rozstrzygnąć pin AccountGenesis, inwentarz
podpisanego wydania i autoryzowany podpis. Kolejny odrębny dowód wymaga
natywnej właściwej usługi, jej odmów i rzeczywistego PostgreSQL. Samo
udostępnienie klienta bibliotecznego nie wykonuje tych kroków.

Nie zmienia się `deployment/stage9_current_status.json`, istniejącego
kontraktu pełnej kwalifikacji IPC, digestów, progów coverage ani bramek CI.
`RUNTIME_IPC_APPLICATION_ADAPTER_UNAVAILABLE` pozostaje w istniejącym
kontrakcie: transport biblioteczny jest przygotowany i testowany, ale
integracja właściwego runtime pozostaje `BLOCKED`. Nie przedstawia się
go jako dostępnego, zainstalowanego klienta aplikacji.
`DURABLE_CHALLENGE_REPLAY_PROTECTION_UNAVAILABLE` pozostaje aktywne;
adapter nie dodaje trwałego magazynu replay. Nie powstają poświadczenia,
konta, klaster, Root Proof ani instalacja na Dellu. Brak natywnego dowodu
właściwej usługi nadal blokuje pełny PASS i gotowość produkcyjną.
