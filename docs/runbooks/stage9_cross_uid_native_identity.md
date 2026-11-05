# Stage 9 — natywna odmowa klienta między UID w jednorazowej VM

Status: **NOT_RUN — brak zatwierdzonej VM w sesji przygotowania**.
Zakres: rzeczywista odmowa niezmienionego `RuntimeFreshnessVerifierClient`
przed wysłaniem ramki, gdy `/proc/PID/exe` obcego UID zwraca EACCES/EPERM.
Próba używa dwóch rzeczywistych UID/GID, systemd, AF_UNIX, SO_PEERCRED
oraz socket-bound SO_PEERPIDFD. Nie zastępuje
[pełnej akceptacji VM](STAGE9_FRESHNESS_SERVICE_VM_ACCEPTANCE.md).

[Harness](../../scripts/diagnostics/stage9_cross_uid_identity.py) nie tworzy
kont, jednostek, instalacji, kluczy, capabilities, bazy ani zatwierdzenia VM.
Po istniejącej zgodzie administrator może utworzyć manifest próby i przejąć
dowody do chronionego katalogu. To jednorazowe narzędzie operatora, bez nowej
usługi uprzywilejowanej lub brokera.
Listener tylko liczy odebrane bajty. Nie uruchamia verifiera, SQL ani operacji
przygotowania. `TEST_ONLY_NATIVE_DENIAL=PASS` dotyczy wyłącznie rzeczywiście
wykonanej odmowy i zera bajtów. Parser zwraca tylko `FORMAT_CONSISTENT/NOT_RUN`;
`postgresql_no_preparation_proof=NOT_RUN` i `product_qualification=BLOCKED`
pozostają jawne. Brak operacji przygotowania w listenerze wyklucza jej
wywołanie w tej próbie, ale nie dowodzi stanu rzeczywistego PostgreSQL.

## Zatwierdzenie i bezpieczny zakres

Wykonać dopiero po osobnym zatwierdzeniu operatora dla jednorazowej Ubuntu
Server 24.04 amd64, CPython 3.12 i systemd jako PID 1. Obowiązuje
[istniejąca bramka VM](STAGE9_LINUX_VM_HARNESS.md#3-zatwierdzenie-vm-przed-execute-vm-trial):
konkretny obraz, snapshot, machine/boot ID, commit, environment-id i wersje
pakietów; sieć i udziały hosta odłączone, bez kluczy produkcyjnych, bez Della.
Kontener lub niekompletna bramka oznacza `BLOCKED`.

Operator dostarcza niezależnie przejrzaną, root-owned kopię dokładnego commitu
w `/srv/cryptohunter-trusted` i sprawdza jej podpisany wykaz hashy przed
uruchomieniem kodu jako root. Nie uruchamiać root z edytowalnego checkoutu.
Wykonywanie poniższych komend jest odrębną czynnością administracyjną operatora,
nie uprawnieniem udzielonym przez ten dokument. Nie uruchamiać ich na komputerze
użytkownika ani fizycznym Dellu.

Próba wymaga osobno zatwierdzonego, chronionego runtime TEST_ONLY CPython3.12
z importowalnymi `cryptography` i `psycopg` zgodnie z zatwierdzonym wheelhouse.
Jego ścieżkę operator zapisuje jako `trial_python`; nie pobiera bibliotek po
izolacji VM i nie zmienia systemowego Pythona. Ten runtime i kopia źródeł są
narzędziami próby, bez deklaracji zatwierdzenia produkcyjnego wydania lub zmiany
zamrożonych pinów. Systemowy Python służy odczytowej bramce środowiska.

Po przygotowaniu approval, przed poniższymi mutacjami:

```sh
cd /srv/cryptohunter-trusted
trial_revision=$(git rev-parse HEAD)
trial_environment_id=noble-cross-uid-001
/usr/bin/python3.12 -s -B -m scripts.diagnostics.stage9_cross_uid_identity preflight \
  --revision "$trial_revision" --environment-id "$trial_environment_id"
```

Wymagany exit 0 i kompletne fakty istniejącego `require_approved_vm`.
Nie tworzyć approval automatycznie na podstawie wyniku inspekcji. Po zmianie
bootu, pakietów lub commitu ponownie uzyskać niezależne zatwierdzenie.

## Ręczne przygotowanie czystego snapshotu

Operator najpierw sprawdza brak obu kont, grup, jednostki produkcyjnej,
`/usr/lib/cryptohunter`, `/run/cryptohunter` i `/var/lib/stage9-cross-uid`.
Zastane zasoby oznaczają przerwanie i nową pustą VM; niczego nie adoptować,
nie usuwać ani nie nadpisywać. Nie używać snapshotu działającego produktu.
Istniejący plik approval w `/etc/cryptohunter` jest dopuszczalny.

Dopiero po tych kontrolach operator tworzy nowe zasoby:

```sh
groupadd --system freshness_verifier_ipc
groupadd --system os_freshness_runtime
useradd --system --no-create-home --shell /usr/sbin/nologin \
  --gid freshness_verifier_ipc os_freshness_crypto_verifier
useradd --system --no-create-home --shell /usr/sbin/nologin \
  --gid os_freshness_runtime --groups freshness_verifier_ipc os_freshness_runtime
install -d -m 0755 -o root -g root /run/cryptohunter
install -d -m 0750 -o os_freshness_crypto_verifier -g freshness_verifier_ipc \
  /run/cryptohunter/freshness
install -d -m 0750 -o root -g os_freshness_runtime /var/lib/stage9-cross-uid
```

Zachować `getent passwd`, `getent group`, `id` obu kont i numery UID/GID;
wymagane dodatnie różne UID oraz różny podstawowy GID runtime i verifiera.
Konto runtime ma wyłącznie swoją podstawową grupę i grupę IPC. Verifier ma
wyłącznie grupę IPC. Wszystkie przodki socketu są root-owned bez zapisu
grupy/innych, ostatni katalog verifier:IPC/0750, bez dodatkowych ACL.
Katalog bazowy dowodów ma root:os_freshness_runtime/0750: tylko runtime
potrzebuje jego odczytu i przeszukiwania (`r+x`) do odczytania snapshotu.
Ten sam wymóg obejmuje wszystkie katalogi ścieżki `before.json`, w tym
opublikowany katalog próby. Verifier nie otrzymuje dostępu do tej ścieżki.
Nie zmienia to publicznej konfiguracji
jednostki root:root/0644 ani przodka `/run/cryptohunter`/0755 bez wrażliwych danych.
Przed publikacją manifestu operator sprawdza też, że oba konta mają odczyt
i przeszukiwanie (`r+x`) każdego katalogu jego łańcucha od `/`, w tym
`/etc/cryptohunter`, bez prawa zapisu i dodatkowych ACL. Samo `x` pozwalające
otworzyć znany plik nie wystarcza: czytnik otwiera katalogi przez
`O_RDONLY | O_DIRECTORY | O_NOFOLLOW`, więc wymaga również `r`.
Brak któregokolwiek prawa oznacza przerwanie próby; nie zmieniać uprawnień
zastanego katalogu ani nie przełączać czytnika na inny tryb otwarcia.
Udany odczyt administratora nie potwierdza praw kont nieuprzywilejowanych;
ich rzeczywiste wywołania i osobna próba VM pozostają wymagane.

Dopiero po utworzeniu i sprawdzeniu powyższych kont oraz grup operator
zatwierdza jednorazowy `trial_id` (UUID lowerhex) i zakres
`TEST_ONLY_STAGE9_CROSS_UID_DENIAL`. `prepare-trial` rozwiązuje grupę IPC
przez NSS, więc nie może poprzedzać przygotowania kont. Po kontroli całej
kopii commitu i runtime, tym samym `trial_python` co w unicie:

```sh
trial_python=/ABSOLUTNA/ZATWIERDZONA/SCIEZKA/python3.12
trial_id=ZATWIERDZONY_UUID_LOWERHEX
"$trial_python" -s -B -m scripts.diagnostics.stage9_cross_uid_identity prepare-trial \
  --revision "$trial_revision" --environment-id "$trial_environment_id" \
  --trial-id "$trial_id" --source-root /srv/cryptohunter-trusted \
  --python-executable "$trial_python"
```

Komenda wymaga UID 0 i oryginalnego `require_approved_vm`. Publikuje przez
`O_EXCL` nowy `/etc/cryptohunter/stage9-cross-uid-trial.json`,
root:freshness_verifier_ipc/0640, czytelny dla obu kont próby, bez dostępu
innych użytkowników. Nowy plik powstaje jako root-owned/0600; początkowy GID
może być odziedziczony z rodzica SGID. Dopiero
po pełnym zapisie, `fsync`, kontroli inode i braku ACL następują
`fchown(0, IPC_GID)` oraz `fchmod(0640)`. Nie ma etapu odczytu dla innych
użytkowników ani zapisu grupy. Zastany plik oznacza odmowę bez adopcji.
Komenda nie tworzy ani nie rozszerza oryginalnej zgody.
Manifest wiąże scope, trial ID, revision/environment, machine/boot, obraz,
snapshot i pakiety z approval, jego dokładny hash, źródła obu skryptów oraz
interpreter. Operator weryfikuje wartości. Dwa hashe skryptów nie dowodzą
pochodzenia całego drzewa importów; wymagana jest niezależna kontrola kopii.

Listener i probe przed bind/connect czytają manifest przez chroniony łańcuch
FD-relative, bez symlinków/hardlinków i zapisu grupy/innych. Sprawdzają bieżące
publiczne fakty VM i izolację, machine/boot/pakiety, rewizję/środowisko z CLI,
skrypty i interpreter. Nie obchodzą wymogu UID 0 istniejącego rootguard.
Odczyt zatwierdzenia i otwieranie chronionych katalogów używają wspólnego
predykatu: każdy katalog, także `/`, musi należeć do root, nie dopuszczać
zapisu grupy/innych i nie mieć POSIX access/default ACL. Niedostępność
obserwacji xattr daje odmowę. Kontrola katalogu poprzedza przejście do
następnego komponentu oraz odczyt danych przez loader; nie wykonuje się
automatycznej naprawy praw ani ACL.
Snapshot/run-probe/compare dodatkowo ponownie sprawdzają oryginalną zgodę
administratora i jej hash. Brak lub niezgodność dowodu oznacza BLOCKED.
Wycofanie upoważnienia wymaga wycofania manifestu przez operatora; nowy boot
lub zmiana źródeł/pakietów unieważnia go automatycznie.

Nowy, wyłącznie testowy plik
`/etc/systemd/system/cryptohunter-freshness-verifier.service` ma root:root/0644.
Nazwa jest celowo dokładna, bo niezmieniony klient wymaga tej cgroup.
Nie jest to jednostka produkcyjna i nie przechodzi jej kwalifikatora.
W `ExecStart` zastąpić poniższą ścieżkę **zatwierdzoną rzeczywistą ścieżką**
`trial_python`; systemd nie interpoluje zmiennej powłoki:

```ini
[Unit]
Description=Stage 9 TEST_ONLY licznik bajtow odmowy cross-UID

[Service]
Type=simple
User=os_freshness_crypto_verifier
Group=freshness_verifier_ipc
WorkingDirectory=/srv/cryptohunter-trusted
ExecStart=/ABSOLUTNA/ZATWIERDZONA/SCIEZKA/python3.12 -s -B -m scripts.diagnostics.stage9_cross_uid_identity listener
UMask=0027
Restart=no
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
ProtectSystem=strict
ReadWritePaths=/run/cryptohunter/freshness
ProtectHome=yes
PrivateTmp=yes
RestrictAddressFamilies=AF_UNIX
```

Brak `ExecStartPre` produkcyjnego jest zamierzony: fixture nie przyjmuje
zatwierdzenia produkcji i nie podmienia podpisanej instalacji. Socket powstaje
atomowo jako 0660 przez lokalny umask0117 jednowątkowego listenera, bez przejściowego chmod po bind. Nie dodawać
`CAP_SYS_PTRACE`, `PR_SET_PTRACER`, wyłączeń Yama, setuid ani zmian procfs.

Do `ExecStart` po `listener` operator dopisuje literalnie:
`--manifest /etc/cryptohunter/stage9-cross-uid-trial.json --revision <zatwierdzone40hex> --environment-id <zatwierdzone-environment-id>`.
Brak argumentu lub manifestu zatrzymuje listener przed bind. Zmienne powłoki
i znaczniki `<...>` nie mogą pozostać w rzeczywistym unicie.

## Wykonanie i sprawdzenie dowodów

Poniższe kroki operator wykonuje w tej samej zatwierdzonej VM i tym samym boocie.
`trial_python` musi wskazywać ten sam runtime co testowy `ExecStart`.
Listener czeka najwyżej 120 s na każde z dwóch połączeń; przygotować polecenie
run-probe przed startem. Po zapisaniu licznika proces pozostaje żywy do
jawnego zatrzymania, aby MainPID/pidfd były dostępne przy końcowej kontroli.

```sh
cd /srv/cryptohunter-trusted
trial_python=/ABSOLUTNA/ZATWIERDZONA/SCIEZKA/python3.12
systemctl daemon-reload
systemctl start cryptohunter-freshness-verifier.service
trial_socket_ready=0
for trial_attempt in 1 2 3 4 5 6 7 8 9 10
do
  if test -S /run/cryptohunter/freshness/verifier.sock
  then
    trial_socket_ready=1
    break
  fi
  sleep 1
done
test "$trial_socket_ready" -eq 1
"$trial_python" -s -B -m scripts.diagnostics.stage9_cross_uid_identity run-probe \
  --revision "$trial_revision" --environment-id "$trial_environment_id"
"$trial_python" -s -B -m scripts.diagnostics.stage9_cross_uid_identity compare \
  --revision "$trial_revision" --environment-id "$trial_environment_id"
```

Nie kontynuować po niezerowym kodzie dowolnego kroku.
Oczekiwanie na socket służy tylko synchronizacji Type=simple, nie dowodzi
tożsamości ani gotowości protokołu. Run-probe czeka najwyżej 5 s na kompletny
JSON licznika po zakończeniu klienta; nigdy nie ponawia wymiany. Jeśli listener nie opublikował
kompletnego licznika, zachować brak dowodu jako `BLOCKED`; sprawdzić jego journal,
a nową próbę wykonać na świeżym klonie snapshotu. Nie zastępować raportu ręcznie.

Run-probe tworzy nowy `/var/lib/stage9-cross-uid/<trial_id>` jako
root:os_freshness_runtime/0700 i utrzymuje deskryptor tego katalogu. Zastany
katalog lub częściowe dowody oznaczają odmowę bez adopcji/recovery. Najpierw
zbiera `before.json`. Ten nowy plik powstaje przez `O_EXCL` jako root-owned/0600;
po pełnym zapisie, `fsync`, sprawdzeniu inode i braku ACL uzyskuje
root:os_freshness_runtime/0640. Verifier nie potrzebuje odczytu snapshotu.
Dopiero po ukończeniu pliku harness publikuje katalog jako 0750 przez
utrzymywany deskryptor, udostępniając snapshot wyłącznie runtime.

W tym oknie sam uruchamia ustalony probe przez `/usr/sbin/runuser` jako runtime
ze stałym argv/env/cwd i przechwytuje stdout przez PIPE (64 KiB, 90 s,
wymagany exit 0). W `finally` blokuje nowe otwarcia przez ścieżkę przez powrót do
0700, także przy wyjątku, timeout albo błędzie klienta. Zamknięcie następuje
przed przejęciem pozostałych dowodów. Dopiero potem kolektor przejmuje stały
licznik listenera i zbiera `after.json`. `client.json`, `listener.json`,
`after.json` i `execution.json` mają root:root/0600. `before.json` pozostaje
0640 wewnątrz zamkniętego katalogu 0700, więc żadne konto nieuprzywilejowane
nie może go ponownie otworzyć. Zamknięcie katalogu nie odbiera deskryptora
wcześniej otwartego przez klienta; klient próby musi być już zakończony lub
zatrzymany po timeout. Zachowany FD katalogu może nadal ujawniać nazwy nowych
wpisów, lecz treść pozostałych plików chroni tryb 0600.

Harness nie przyjmuje cudzych raportów do „opieczętowania”. `execution.json`
powstaje przez `O_EXCL` dopiero po walidacji wykonania i wiąże dokładne hashe
wszystkich czterech wejść, manifest oraz argv. PID runuser nie jest utożsamiany
z PIDem Pythona klienta.

Przed publikacją manifestu, snapshotu lub katalogu harness odmawia przy
`system.posix_acl_access` albo `system.posix_acl_default` w całym sprawdzanym
łańcuchu katalogów od `/` lub nowym częściowym obiekcie. Kontrola
nowego obiektu w mechanizmie publikacji poprzedza zwiększenie trybu:
odziedziczona ACL z początkowo zerową maską mogłaby uzyskać odczyt po `chmod`.
Harness nie usuwa ACL, nie naprawia przodków i nie zmienia uprawnień zastanych
obiektów. Właściciel zapisów pozostaje root; grupa ma wyłącznie wskazany odczyt
lub przejście, a inni użytkownicy nie otrzymują dostępu do dowodów.

Kontrakt odczytu istniejącego końcowego pliku nadal sprawdza zwykły typ,
właściciela root, brak zapisu grupy/innych, pojedynczy hardlink, ograniczony
rozmiar i stabilność metadanych podczas odczytu. Wspólny predykat ACL dotyczy
katalogów; nie ustanawia ogólnej kontroli ACL wszystkich istniejących plików
ani executable. Kontrola ACL nowego dowodu pozostaje odrębnym warunkiem
mechanizmu publikacji przed jego udostępnieniem.

Root snapshot odczytuje prawdziwy `/proc/MainPID/exe`; to wejście wyłącznie
TEST_ONLY z niezależnego obserwatora, nie akceptacja kodu usługi. Snapshot nie
jest odpowiedzią listenera. Odczyty systemd są rzeczywiste i wykonywane także
przez runtime, ale nie poświadczają ciągłości obrazu.

Probe otwiera pierwsze połączenie wyłącznie do odczytów SO_PEERCRED/SO_PEERPIDFD,
procfs i dokładnych kontroli istniejącego transportu. Wymaga, aby wszystkie
obserwacje poza executable odpowiadały root snapshotowi, a `/proc/PID/exe`
rzeczywiście zwracało EACCES/EPERM i `exe_sha256=None`. Wywołuje istniejący
`_check_server`, wymaga konkretnej odmowy, następnie na drugim połączeniu
uruchamia niezmieniony `RuntimeFreshnessVerifierClient.verify_exact_candidate`.
Wokół tego jednego rzeczywistego wywołania stosuje obserwacyjny `sys.settrace`:
wymaga niezajętego rejestratora, rozpoznaje dokładną tożsamość obiektu
`_check_server.__code__`, zapisuje jedno wejście z kopią rzeczywistego
`server.process` oraz jeden konkretny `PermissionError`. Nie zmienia
locals/globals, argumentów, wyniku ani wyjątku, a w `finally` usuwa rejestrator.
Zapis musi odpowiadać wcześniejszemu procesowi, z brakującym exe. Wcześniejsza
odmowa wewnątrz PinnedReadOnlyPeer nie ma takiego zapisu i daje `BLOCKED`.
Rejestrator sam nie poświadcza jądra; dowodami towarzyszącymi pozostają
rzeczywiste errno procfs, SO_PEERPIDFD i licznik bajtów oddzielnego listenera.
Po rzeczywistym wywołaniu ponownie odczytuje proces, tożsamość runtime i
konkretne errno executable; drift daje `BLOCKED`. Wywołania systemctl mają
stałe PATH/LANG/LC_ALL bez dziedziczenia adresu D-Bus. Nie zastępuje UID, PID,
executable, socketu, metod transportu ani wyniku klienta.
Syntetyczne wejście `b"{}"` nie opuszcza klienta. Listener nie zna żadnego
sekretu, podpisu ani dostępu SQL.

`compare` wymaga aktualnej zgody administratora, manifestu, chronionego
`execution.json`, zgodnych bajtów przejętych artefaktów oraz świeżej obserwacji
zgodnej ze snapshotami przed/po. Sprawdza machine/boot, revision/environment,
obraz, MainPID/UID/GID, okres jednostki i cały rekord procesu; sam hash
nieużytego załącznika nie wystarcza. Wymaga także dwóch zakończonych EOF
połączeń od rzeczywistego PID/UID/GID
probe, **0 odebranych bajtów na obu**, zgodnego bootu i listenera, odmowy procfs
oraz wyniku klienta `UNAVAILABLE/TRANSPORT_OR_IDENTITY_UNAVAILABLE` i dokładnego
zapisu wywołania oraz wyjątku kontroli serwera podczas tego wywołania klienta.
Wcześniejsza odmowa ścieżki, UID, grup, cgroup, D-Bus/properties albo brak
SO_PEERPIDFD nie zalicza testu. Inny UID, brak wymaganych pól lub dostępne exe
oznaczają `BLOCKED`, a nie sztuczny sukces negatywny.

Zachować dokładne kody wyjścia, oba raporty i snapshot, stdout porównania,
journal jednostki, systemd properties, wersje jądra/systemd/Pythona, mountinfo
procfs, metadane/ACL socketu i katalogów oraz zewnętrzny identyfikator VM/obrazu.
To dowody testu w zaufanej VM, nie podpisana atestacja ani autorytet produktu.
Root jest zaufanym kolektorem. Kontrola dostępu i hashe chronią przejęte
artefakty przed późniejszą podmianą przez runtime/verifiera; nie chronią
przed złośliwym root ani przejętym w pamięci autorem raportu. Trial ID nie jest
sekretem ani uwierzytelnieniem. Hash obrazu pochodzi z zewnętrznej zgody
operatora, nie z lokalnej atestacji hypervisora.
Po ich zebraniu operator jawnie zatrzymuje wyłącznie własną jednostkę testową
`systemctl stop cryptohunter-freshness-verifier.service`, eksportuje dowody
odłączanym nośnikiem i usuwa całą jednorazową VM. Nie czyści ręcznie produktu.

## Osobna kwalifikacja dostępu w VM — NOT_RUN

Poniższe sprawdzenie wymaga osobno zatwierdzonego scenariusza operatora z
rzeczywistymi UID/GID w jednorazowej VM. Nie wykonano go w tej sesji. Lokalne
regresje modelowane sprawdzają kolejność operacji i odmowy; nie dowodzą
natywnych uprawnień między kontami.

| Obiekt i etap | Runtime | Verifier | Trzecie konto bez grup runtime/IPC |
| --- | --- | --- | --- |
| Manifest root:IPC/0640 | Odczyt dozwolony | Odczyt dozwolony | Odmowa odczytu |
| Ukończony `before.json`, katalog udostępniony jako 0750 | Odczyt dozwolony | Odmowa odczytu | Odmowa odczytu |
| `before.json` i pozostałe dowody po zamknięciu katalogu do 0700 | Odmowa nowego otwarcia | Odmowa odczytu | Odmowa odczytu |

Operator niezależnie sprawdza NSS i rzeczywiste grupy trzeciego konta, np.
`nobody`; sama nazwa konta nie dowodzi braku członkostwa w grupach. Zbiera
kody powodzenia otwarcia lub konkretne EACCES/EPERM oraz właścicieli, tryby
i brak ACL, bez drukowania treści dowodów. W osobnym scenariuszu publikacji
używa niesekretnego znacznika TEST_ONLY i tych samych helperów uprawnień,
aby obserwować także okno 0750 oraz zamknięcie po wyjątku lub timeout.
Nie dodaje opcji pomijania bramki, sztucznego opóźnienia ani rozszerzenia
dostępu w narzędziu odmowy IPC. Brak obserwacji któregoś etapu pozostaje
`NOT_RUN/BLOCKED`; zgodność samych trybów lub testy pytest nie zaliczają
kwalifikacji rzeczywistych kont.

## Próby pozostające poza tym harnessem

Pełna dodatnia wymiana rzeczywistego verifiera z zatwierdzonym materiałem
TEST_ONLY, niezależny licznik `prepared_verifications` PostgreSQL przed/po,
macierz execve/execveat/fork/listener/SCM_RIGHTS, restart podczas wymiany
oraz egzekwowanie ciągłości wymagają osobnego planu i zatwierdzenia VM.
Ten harness nie dostarcza dodatniego wariantu architektury. Nie rozwiązuje
AccountGenesis, replay ani zamrożonych digestów; wszystkie istniejące blokady
pozostają aktywne, `STAGE9_PRODUCTION_READY=NO`, **bez merge**.

## Lokalne testy integralności raportu

Tryb `format-reports client.json listener.json before.json` jest wyłącznie
parserem i może działać poza VM. Jego jedynym dodatnim wynikiem jest
`FORMAT_CONSISTENT`, z `native_status=NOT_RUN`, nigdy PASS. Nie zastępuje
natywnego compare i nie zapisuje receipt. Syntetyczne JSON nie mogą same
awansować do natywnego wyniku.

`tests/scripts/test_stage9_cross_uid_identity_reports.py` sprawdza format
i odmowy zmian bindingów, snapshotu, systemd, procesu oraz kontroli klienta.
`test_stage9_cross_uid_identity_gates.py` i `test_stage9_cross_uid_trial.py`
sprawdzają kolejność odmów i manifest, w tym brak receipt.
`test_stage9_cross_uid_permissions.py` sprawdza modelowane regresje publikacji
uprawnień i odmów ACL. Te testy nie uruchamiają klienta produktu ani usługi,
nie zmieniają UID/PID/executable i nie stanowią wyniku natywnej próby.
`test_stage9_cross_uid_directory_contract.py` dodatkowo porównuje odmowy obu
przejść po katalogach przy braku `r`, ACL przodka i braku obserwacji xattr. Jego
osobna sonda Linux potwierdza na własnym katalogu tymczasowym różnicę między
bezpośrednim otwarciem pliku a odczytowym otwarciem katalogu; używa jednego rzeczywistego UID.
Przy root lub poza Linux jest jawnie pomijana. Nie dowodzi izolacji dwóch
kont, systemd ani kwalifikacji VM.
Wykonywać osobno:

```sh
python -m pytest --noconftest -q \
  tests/scripts/test_stage9_cross_uid_identity_reports.py \
  tests/scripts/test_stage9_cross_uid_identity_gates.py \
  tests/scripts/test_stage9_cross_uid_trial.py \
  tests/scripts/test_stage9_cross_uid_permissions.py \
  tests/scripts/test_stage9_cross_uid_directory_contract.py
```
