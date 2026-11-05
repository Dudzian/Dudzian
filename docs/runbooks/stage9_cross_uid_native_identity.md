# Stage 9 — natywna odmowa klienta między UID w jednorazowej VM

Status: **NOT_RUN — brak zatwierdzonej VM w sesji przygotowania**.
Zakres: rzeczywista odmowa niezmienionego `RuntimeFreshnessVerifierClient`
przed wysłaniem ramki, gdy `/proc/PID/exe` obcego UID zwraca EACCES/EPERM.
Próba używa dwóch rzeczywistych UID/GID, systemd, AF_UNIX, SO_PEERCRED
oraz socket-bound SO_PEERPIDFD. Nie zastępuje
[pełnej akceptacji VM](STAGE9_FRESHNESS_SERVICE_VM_ACCEPTANCE.md).

[Harness](../../scripts/diagnostics/stage9_cross_uid_identity.py) nie tworzy
kont, jednostek, instalacji, kluczy, capabilities, bazy ani zatwierdzenia.
Listener tylko liczy odebrane bajty. Nie uruchamia verifiera, SQL ani operacji
przygotowania. Wynik `PASS` dotyczy wyłącznie odmowy i zera bajtów;
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
install -d -m 0755 -o root -g root /var/lib/stage9-cross-uid
```

Zachować `getent passwd`, `getent group`, `id` obu kont i numery UID/GID;
wymagane dodatnie różne UID oraz różny podstawowy GID runtime i verifiera.
Konto runtime ma wyłącznie swoją podstawową grupę i grupę IPC. Verifier ma
wyłącznie grupę IPC. Wszystkie przodki socketu są root-owned bez zapisu
grupy/innych, ostatni katalog verifier:IPC/0750, bez dodatkowych ACL.

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

## Wykonanie i sprawdzenie dowodów

Poniższe kroki operator wykonuje w tej samej zatwierdzonej VM i tym samym boocie.
`trial_python` musi wskazywać ten sam runtime co testowy `ExecStart`.
Listener czeka najwyżej 45 s na każde z dwóch połączeń; przygotować polecenia
snapshot/probe przed startem. Po zapisaniu licznika proces pozostaje żywy do
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
test ! -e /var/lib/stage9-cross-uid/expected.json
"$trial_python" -s -B -m scripts.diagnostics.stage9_cross_uid_identity snapshot \
  --revision "$trial_revision" --environment-id "$trial_environment_id" \
  > /var/lib/stage9-cross-uid/expected.json
chmod 0644 /var/lib/stage9-cross-uid/expected.json
runuser -u os_freshness_runtime -- "$trial_python" -s -B \
  -m scripts.diagnostics.stage9_cross_uid_identity probe \
  /var/lib/stage9-cross-uid/expected.json \
  > /var/lib/stage9-cross-uid/client.json
"$trial_python" -s -B -m scripts.diagnostics.stage9_cross_uid_identity compare \
  /var/lib/stage9-cross-uid/client.json \
  /run/cryptohunter/freshness/native-listener.json
```

Nie kontynuować po niezerowym kodzie dowolnego kroku; chmod wykonać wyłącznie
po sukcesie snapshotu. Snapshot musi być nowy, root-owned i czytelny dla runtime.
Oczekiwanie na socket służy tylko synchronizacji Type=simple, nie dowodzi
tożsamości ani gotowości protokołu. Porównanie czeka najwyżej 5 s na kompletny
JSON licznika; nigdy nie ponawia wymiany. Jeśli listener nie opublikował
kompletnego licznika, zachować brak dowodu jako `BLOCKED`; sprawdzić jego journal,
a nową próbę wykonać na świeżym klonie snapshotu. Nie zastępować raportu ręcznie.
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

`compare` wymaga dwóch zakończonych EOF połączeń od rzeczywistego PID/UID/GID
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
Po ich zebraniu operator jawnie zatrzymuje wyłącznie własną jednostkę testową
`systemctl stop cryptohunter-freshness-verifier.service`, eksportuje dowody
odłączanym nośnikiem i usuwa całą jednorazową VM. Nie czyści ręcznie produktu.

## Próby pozostające poza tym harnessem

Pełna dodatnia wymiana rzeczywistego verifiera z zatwierdzonym materiałem
TEST_ONLY, niezależny licznik `prepared_verifications` PostgreSQL przed/po,
macierz execve/execveat/fork/listener/SCM_RIGHTS, restart podczas wymiany
oraz egzekwowanie ciągłości wymagają osobnego planu i zatwierdzenia VM.
Ten harness nie dostarcza dodatniego wariantu architektury. Nie rozwiązuje
AccountGenesis, replay ani zamrożonych digestów; wszystkie istniejące blokady
pozostają aktywne, `STAGE9_PRODUCTION_READY=NO`, **bez merge**.

## Lokalne testy integralności raportu

`tests/scripts/test_stage9_cross_uid_identity_reports.py` sprawdza format bazowy i
odrzucenie dziesięciu spreparowanych raportów: bajt żądania, brak EOF, brak
pidfd, zadeklarowane dostępne exe, ten sam UID, rozszerzenie kwalifikacji,
brak wejścia do kontroli serwera, inny wyjątek, inny rekord procesu i brak trace.
Nie uruchamia klienta ani usługi, nie zmienia UID/PID/executable i nie jest
wynikiem natywnej próby. Wykonywać osobno:

```sh
python -m pytest --noconftest -q tests/scripts/test_stage9_cross_uid_identity_reports.py
```
