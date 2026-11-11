# Stage 9 — jednorazowa próba operatora w Ubuntu 24.04 VM

> **Procedura historyczna v1 — nie wykonywać jej eksportu ani fazy dodatniej
> dla aktualnego wydania.** Poniższa blokada bibliotek i manifest v1 opisują
> profil sprzed podpisanego runtime. Bieżący instalator wymaga
> `cryptohunter.freshness.substrate.v2`, schema_version=2 i chronionego bindingu.
> Aktualna procedura, tryby INSPECT/EXECUTE-VM-TRIAL, eksport v2 i snapshoty są w
> [STAGE9_LINUX_VM_HARNESS.md](STAGE9_LINUX_VM_HARNESS.md). Zachowana treść nie
> stanowi dowodu ani instrukcji wykonania bieżącej instalacji.
>
> Bieżący profil v2 zawiera 16 plików payloadu oraz sześć plików bootstrapu,
> w tym `deployment/freshness_application.py`, i zachowuje podpisany manifest,
> podpis, klucz publiczny oraz binding aplikacji do kontroli przy starcie.
> Historycznych list i eksportu v1 poniżej nie należy uzupełniać ręcznie.
> Nowy pakiet wymaga zatwierdzenia aktualnego PR przed podpisaniem; jego zakres
> minimalny nie stanowi pełnej kwalifikacji `PRODUCTION_LOCAL`.

To instrukcja do ręcznego wykonania w jednorazowej VM. Jej poleceń
uprzywilejowanych nie uruchamia zwykły pytest, CI ani agent przygotowujący PR.
Nie dotyczy fizycznego Della. Nie używać produkcyjnych kluczy, danych, sieci
ani współdzielonych katalogów hosta. Wynik nie awansuje Stage 9.

## Bieżąca blokada i punkt zatrzymania

Obecny profil wskazuje `/usr/bin/python3` zarówno w wrapperze instalatora,
jak i w jednostce verifiera. Ubuntu 24.04 zapewnia Python 3.12, lecz standardowe
pakiety `python3-cryptography` 41.0.7 i `python3-psycopg` 3.1.17 nie spełniają
wymagań projektu `cryptography>=50.0.1,<51` i `psycopg>=3.2,<4`.
Źródła, sonda importów CPython 3.12 oraz projekt osobnego runtime są opisane
w [raporcie remediacji](../architecture/stage9_linux_installer_remediation.md).

Na standardowym Noble wykonalna jest poniższa faza przygotowania i rzeczywista
odmowa preflight. **Dodatnia instalacja pozostaje BLOCKED. Zatrzymać się przed
fazą dodatnią**, dopóki osobno zatwierdzony kontrakt runtime nie zostanie
zaimplementowany, przetestowany i związany z wydaniem. Nie wystarczy lokalnie
zmienić ExecStart na venv. Nie stosować systemowego pip, `--break-system-packages`,
PYTHONPATH, `.pth`, podmiany `/usr/bin/python3` ani obniżania wymagań.

Instrukcje fazy dodatniej określają zamknięty plan kolejnej próby po usunięciu
tej blokady. Nie stanowią dowodu, że obecny profil działa na standardowym Ubuntu.

## 1. VM i zachowanie dowodów

Operator tworzy VM ręcznie w hypervisorze, z Ubuntu Server 24.04 amd64,
natywnym systemd jako PID 1 i konsolą umożliwiającą odzyskiwanie. Nie używać
kontenera jako zamiennika tej próby. Wyłączyć shared folders, clipboard,
przekazywanie urządzeń i dostęp do Della. Nośnik wejściowy może być odrębnym,
odłączanym obrazem ISO tylko do odczytu; nie montować filesystemu hosta.

Zachować identyfikator obrazu ISO, SHA-256, dowód weryfikacji podpisanych
sum Ubuntu, konfigurację VM i zrzut konsoli. Po wyłączeniu VM zachować hash
jej bazowego dysku i identyfikatory snapshotów. Nie haszować aktywnie zmienianego
dysku jako niezmiennego obrazu. Utworzyć snapshot `S0-clean` przed przygotowaniem.

W konsoli gościa zapisać do własnego katalogu dowodów:

```sh
mkdir -m 0700 stage9-vm-evidence
trial_evidence=$(realpath stage9-vm-evidence)
cat /etc/os-release > "$trial_evidence/os-release.txt"
uname -a > "$trial_evidence/uname.txt"
cat /proc/1/comm > "$trial_evidence/pid1.txt"
systemctl --version > "$trial_evidence/systemd-version.txt"
/usr/bin/python3 --version > "$trial_evidence/python-version.txt" 2>&1
```

PID 1 musi być `systemd`, system Ubuntu 24.04, a interpreter Python 3.12.
Przy innym wyniku zakończyć próbę. Odczyty to obserwacja obrazu VM, nie dowód
zaufania do produkcyjnego hosta.

## 2. Oficjalne pakiety i osobny klaster fixture

Poniższe mutacje wykonuje wyłącznie operator w tej VM. APT i podpisy repozytoriów
pozostają włączone; zachować listę źródeł oraz wersje kandydatów. Dostęp sieciowy
może służyć instalacji oficjalnych pakietów. Po przygotowaniu odłączyć sieć.

```sh
sudo apt-get update
sudo apt-get install --no-install-recommends postgresql-common
```

Przed instalacją serwera operator przez `sudoedit` ustawia w
`/etc/postgresql-common/createcluster.conf` dokładnie
`create_main_cluster = false`. Sprawdzić obowiązującą składnię dostarczonego
pakietu i zachować kopię przed/po. Instalacja APT nie ma tworzyć nieoczekiwanego
klastra automatycznie. Następnie:

```sh
sudo apt-get install --no-install-recommends \
  python3 python3-venv python3-cryptography python3-psycopg \
  postgresql-16 postgresql-client-16 systemd dbus openssl strace
apt-cache policy python3 python3-cryptography python3-psycopg postgresql-16 \
  > "$trial_evidence/apt-policy.txt"
pg_lsclusters --no-header > "$trial_evidence/clusters-before.txt"
```

Lista klastrów powinna być pusta. Przy zastanym klastrze zatrzymać próbę;
nie usuwać ani nie adoptować go automatycznie. Zachować snapshot `S1-packages`.
To także punkt odtworzenia przypadku „pakiet PostgreSQL jest, klastra brak”.

Dopiero po osobnym zatwierdzeniu operator jawnie tworzy pusty klaster fixture:

```sh
sudo pg_createcluster 16 main --port=5432
sudo systemctl start postgresql@16-main.service
pg_lsclusters --no-header > "$trial_evidence/clusters-created.txt"
systemctl show postgresql@16-main.service --property=LoadState,ActiveState \
  > "$trial_evidence/postgresql-unit.txt"
```

Oczekiwany jest wyłącznie `16 main 5432 online postgres` oraz załadowana,
aktywna jednostka. Wymaganą dyrektywę `include_dir = 'conf.d'` sprawdza
odczytowy preflight aktualnego instalatora, z przejrzanego i chronionego
bootstrapu, zgodnie z [procedurą v2](STAGE9_LINUX_VM_HARNESS.md#6-wejścia-i-kolejne-fazy-wykonania).
Nie używać `pg_conftool 16 main show include_dir` jako kryterium akceptacji:
poprawna konfiguracja Noble daje tu kod 1 i pusty wynik, ponieważ narzędzie
rozwija include zamiast zwracać tę dyrektywę jako parametr. Szczegóły i granice
kontroli opisuje [remediacja host-probe](../architecture/stage9_linux_installer_remediation.md#remediacja-host-probe-postgresql--2026-11-11).
Własność `postgres` katalogu `conf.d` może być poprawna dla Ubuntu, lecz
bieżący instalator wymaga root-owned `0755`. Nie zmieniać tych metadanych
na podstawie historycznej procedury; zgłoszony ServiceTrial pozostaje
`BLOCKED` do [osobnej oceny operatora](STAGE9_HOST_PROBE_REMEDIATION_VM_PLAN.md).
Nie wykonywać tych poleceń na Dellu. Nie używać `pg_dropcluster` w recovery.

## 3. Dane zastane, backup i konta

Przed instalacją aplikacji utworzyć niezależny, syntetyczny znacznik danych.
Nie zastępować nim produkcyjnego schematu Freshness ani nie otwierać jej bramki:

```sh
sudo -u postgres createdb stage9_vm_guard
sudo -u postgres psql -X -v ON_ERROR_STOP=1 stage9_vm_guard <<'SQL'
CREATE TABLE installation_guard(id integer PRIMARY KEY, marker text NOT NULL);
INSERT INTO installation_guard VALUES (1, 'stage9-vm-observation');
SQL
sudo -u postgres pg_dump -Fc stage9_vm_guard > "$trial_evidence/guard-before.dump"
pg_restore --list "$trial_evidence/guard-before.dump" \
  > "$trial_evidence/guard-backup-list.txt"
sudo -u postgres psql -X -At stage9_vm_guard \
  -c 'SELECT id,marker FROM installation_guard ORDER BY id' \
  > "$trial_evidence/guard-before.rows"
sha256sum "$trial_evidence/guard-before.rows" > "$trial_evidence/guard-before.sha256"
```

Backup sprawdzić osobno przez odtworzenie do innej pustej bazy fixture w tej VM
i porównanie wierszy. Nie odtwarzać go na oryginalną bazę. Zachować snapshot.

Sprawdzić `getent passwd`/`getent group` dla poniższych nazw i wszystkie
przydziały UID/GID. Jeżeli którekolwiek konto/grupa już istnieje, zatrzymać
próbę przed mutacją i ocenić zastany stan. Dla czystej VM operator wykonuje:

```sh
sudo groupadd --system os_freshness_crypto_verifier
sudo groupadd --system os_freshness_runtime
sudo groupadd --system freshness_verifier_ipc
sudo useradd --system --gid os_freshness_crypto_verifier \
  --groups freshness_verifier_ipc --home-dir /nonexistent --no-create-home \
  --shell /usr/sbin/nologin os_freshness_crypto_verifier
sudo useradd --system --gid os_freshness_runtime \
  --groups freshness_verifier_ipc --home-dir /nonexistent --no-create-home \
  --shell /usr/sbin/nologin os_freshness_runtime
id os_freshness_crypto_verifier > "$trial_evidence/verifier-id.txt"
id os_freshness_runtime > "$trial_evidence/runtime-id.txt"
getent group freshness_verifier_ipc > "$trial_evidence/ipc-group.txt"
```

Oba UID i własne podstawowe GID muszą być różne, z zakresu 1–999; IPC GID jest
odrębny także od `postgres`. Grupa IPC ma dokładnie dwóch wymienionych członków,
własne grupy podstawowe nie mają dodatkowych członków, konta nie mają innych
członkostw ani uprawnień administratora. Sprawdzić także lookup name→id→name
i brak obcego konta z chronionym GID jako grupą podstawową. Nie odczytywać
`shadow` do raportu. Nie zakładać, że usunięcie kont odwraca jego użycie.

## 4. Rootless builder materiałów TEST ONLY

Builder działa bez root na odrębnym stanowisku z przejrzanym checkoutem. Nie
importuje kodu instalatora. Eksportuje wyłącznie Git blobs zatwierdzonego SHA,
odczytując literalny `REQUIRED_FILES` z AST. Obecny inwentarz ma 13 plików;
offline helper kwalifikacji nie jest częścią import closure usługi. Nie dodawać
go tylko po to, by zaczął wyglądać jak kwalifikacja zainstalowanego runtime.

```sh
trial_revision=$(git rev-parse HEAD)
export TRIAL_REVISION="$trial_revision"
python3 -I -B - <<'PY'
import ast, hashlib, json, os, subprocess
from pathlib import Path, PurePosixPath

revision = os.environ['TRIAL_REVISION']
assert len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
def git(*args):
    return subprocess.check_output(['git', *args])
source = git('show', revision + ':deployment/production_local_installer.py')
assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)]
node = next(n.value for n in assignments
            if any(isinstance(t, ast.Name) and t.id == 'REQUIRED_FILES'
                   for t in n.targets))
assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
assert node.func.id == 'frozenset' and len(node.args) == 1 and not node.keywords
required = ast.literal_eval(node.args[0])
assert isinstance(required, set) and all(isinstance(p, str) for p in required)
out = Path('stage9-vm-release')
out.mkdir(mode=0o700)  # Odmowa, jeżeli zastany katalog już istnieje.
files = []
def export(relative, destination):
    path = PurePosixPath(relative)
    assert not path.is_absolute() and str(path) == relative
    assert '..' not in path.parts and path.parts[0] in {'bot_core', 'deployment'}
    record = git('ls-tree', revision, '--', relative).decode('ascii')
    mode = record.split()[0]
    assert mode in {'100644', '100755'}
    data = git('show', revision + ':' + relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    permission = 0o755 if mode == '100755' else 0o644
    destination.chmod(permission)
    return data, permission
for relative in sorted(required):
    data, mode = export(relative, out / 'payload' / relative)
    files.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest(),
                  'size': len(data), 'mode': mode})
manifest = {'schema_version': 1, 'scope': 'cryptohunter.freshness.substrate.v1',
            'source_revision': revision, 'files': files}
(out / 'release.json').write_bytes(json.dumps(
    manifest, sort_keys=True, separators=(',', ':')).encode('utf-8'))
bootstrap = ('deployment/install-production-local.sh',
             'deployment/production_local_installer.py',
             'deployment/production_local_installer_host.py')
sums = []
for relative in bootstrap:
    data, _ = export(relative, out / 'bootstrap' / relative)
    sums.append(hashlib.sha256(data).hexdigest() + '  ' + relative + '\n')
(out / 'bootstrap.sha256').write_text(''.join(sums), encoding='ascii')
(out / 'revision.txt').write_text(revision + '\n', encoding='ascii')
print('TEST ONLY:', revision, len(files), 'plików; materiały jeszcze niezatwierdzone')
PY
```

Drugi operator niezależnie porównuje SHA z przejrzanym PR, sprawdza kod
bootstrapu, wymagany inwentarz, hashe i tryby oraz podpisuje dopiero zatwierdzone
materiały. Podpis własnego nieprzejrzanego bundle nie ustanawia zaufania.
Przykładowa osobna ceremonia z **nowym, jednorazowym kluczem TEST ONLY**:

```sh
umask 077
openssl genpkey -algorithm ED25519 -out vm-test-authority.pem
openssl pkey -in vm-test-authority.pem -pubout -out vm-test-authority.pub.pem
openssl pkey -in vm-test-authority.pem -pubout -outform DER -out vm-test-authority.der
python3 -I -B - <<'PY'
from pathlib import Path
der = Path('vm-test-authority.der').read_bytes()
assert len(der) == 44 and der[:12] == bytes.fromhex('302a300506032b6570032100')
Path('vm-test-authority.raw').write_bytes(der[12:])
PY
openssl pkeyutl -sign -rawin -inkey vm-test-authority.pem \
  -in stage9-vm-release/release.json -out stage9-vm-release/release.sig
openssl pkeyutl -sign -rawin -inkey vm-test-authority.pem \
  -in stage9-vm-release/bootstrap.sha256 -out stage9-vm-release/bootstrap.sig
sha256sum vm-test-authority.raw
```

Klucz prywatny zostaje poza VM, repozytorium i nośnikiem. Nie używać klucza
produkcyjnego. Operator VM otrzymuje odrębnym kanałem zatwierdzony fingerprint
klucza publicznego; samego klucza z nieznanego bundle nie uznaje za autorytet.
Na nośniku przenosi się publiczne PEM/raw, podpisy, manifest, bootstrap i payload.
Nie przepisywać manifestu po podpisaniu.

## 5. Zaufany bootstrap i rzeczywisty preflight

W VM skopiować materiały do nowego katalogu wejściowego bez root, odtworzyć
podpisane tryby plików i odłączyć nośnik. Nie zakładać, że tryby plików na ISO
odpowiadają manifestowi. Nie przyjmować symlinków ani hardlinków. Operator
porównuje fingerprint publicznego raw32 z uzyskanym odrębnie i sprawdza, że PEM
zawiera ten sam klucz (DER ma dokładnie powyższy prefix i 32 bajty wartości).
Następnie sprawdza oba podpisy przez zaufany OpenSSL Ubuntu:

```sh
openssl pkeyutl -verify -rawin -pubin -inkey vm-test-authority.pub.pem \
  -in stage9-vm-release/release.json -sigfile stage9-vm-release/release.sig
openssl pkeyutl -verify -rawin -pubin -inkey vm-test-authority.pub.pem \
  -in stage9-vm-release/bootstrap.sha256 -sigfile stage9-vm-release/bootstrap.sig
(cd stage9-vm-release/bootstrap && sha256sum -c ../bootstrap.sha256)
```

Dopiero po niezależnym przeglądzie instalować trzy dokładne pliki bootstrapu
przez `sudo install` do nowego `/opt/approved-installer/deployment`: root:root,
katalogi0755, wrapper0755, dwa moduły0644. Zatrzymać próbę, jeśli ścieżka istnieje,
ma symlink lub niezgodnego właściciela. Ponownie porównać hashe z podpisanym
`bootstrap.sha256`. Nie wykonywać Pythona jako root z edytowalnego checkoutu.
Publiczny raw32 instalować root:root0644 pod
`/etc/cryptohunter-vm-test-authority.pub`, poza bundle, z chronionymi przodkami.
Nie instalować tam klucza prywatnego. Zachować dowód metadanych i fingerprintu.

Po potwierdzeniu, że docelowe ścieżki nie istnieją i ich przodkowie są
chronieni, operator wykonuje dokładne kopiowanie trzech przejrzanych plików:

```sh
trial_release=$(realpath stage9-vm-release)
sudo install -d -o root -g root -m 0755 /opt/approved-installer/deployment
sudo install -o root -g root -m 0755 \
  "$trial_release/bootstrap/deployment/install-production-local.sh" \
  /opt/approved-installer/deployment/install-production-local.sh
sudo install -o root -g root -m 0644 \
  "$trial_release/bootstrap/deployment/production_local_installer.py" \
  /opt/approved-installer/deployment/production_local_installer.py
sudo install -o root -g root -m 0644 \
  "$trial_release/bootstrap/deployment/production_local_installer_host.py" \
  /opt/approved-installer/deployment/production_local_installer_host.py
sudo install -o root -g root -m 0644 vm-test-authority.raw \
  /etc/cryptohunter-vm-test-authority.pub
(cd /opt/approved-installer && sha256sum -c "$trial_release/bootstrap.sha256")
```

To jawne przygotowanie operatora przed pomiarem preflight, nie operacje
wykonywane przez diagnostykę. Nie powtarzać tych poleceń przeciw zastanemu
bootstrapowi bez ponownej oceny i backupu.

Ustawić własne ścieżki absolutne oraz pełne zatwierdzone SHA, bez symlinków:

```sh
trial_release=$(realpath stage9-vm-release)
trial_revision=$(cat "$trial_release/revision.txt")
sudo /opt/approved-installer/deployment/install-production-local.sh \
  --preflight --bundle "$trial_release/payload" \
  --manifest "$trial_release/release.json" --signature "$trial_release/release.sig" \
  --trusted-key /etc/cryptohunter-vm-test-authority.pub \
  --expected-revision "$trial_revision" \
  > "$trial_evidence/preflight.json"
trial_preflight_rc=$?
printf '%s\n' "$trial_preflight_rc" > "$trial_evidence/preflight.exit"
```

Nie uruchamiać tego bloku w shellu z `set -e`, który zgubi zapis oczekiwanej
odmowy. Na standardowym Noble z bibliotekami APT, poprawnym podpisem i ścieżkami
oczekiwany jest `REFUSED`, kod20, `production_qualification=WITHHELD`.
Bez `cryptography` możliwa jest wcześniejsza odmowa10 podpisu/importu.
Inna wcześniejsza odmowa musi zostać wyjaśniona, nie przepisana jako sukces.

Przed/po zachować odczytowy inwentarz istnienia, typów, UID/GID, trybów i hashy
pod `/usr/lib/cryptohunter`, `/usr/lib/.cryptohunter-stage-*`,
`/var/lib/cryptohunter-installer`, `/etc/cryptohunter`, dokładnym plikiem
jednostki oraz dokładnym plikiem w `conf.d`. Nie podążać za symlinkami;
nie porównywać atime zmienianego samym odczytem. Odmowa preflight ma zachować
identyczny stan tych ścieżek, brak nowego dziennika i brak blokady instalacji.
Porównać także wiersze `installation_guard` i zachować dotychczasowy backup.

**STOP: obecny profil nie ma spełnionych zależności Noble. Nie przechodzić
do `--install` i nie uruchamiać usług.** Zachować wynik jako poprawną odmowę,
nie PASS instalacji. Snapshot `S2-observed-refusal` kończy wykonalną fazę
standardowego obrazu w tym PR.

## 6. Bramka przyszłej fazy dodatniej

Operator może wznowić próbę z nowego, osobno zatwierdzonego snapshotu dopiero,
gdy wdrożono projekt runtime z raportu: niezależny podpis i pełny lock/hash
środowiska, CPython3.12 bez system-site/user-site, chroniony interpreter/importy,
związanie environment-id z manifestem aplikacji, nowy exact ExecStart oraz
sprawdzenie `/proc/MainPID/exe`, argv, PIDFD, UID/GID i cgroup.

Zatwierdzony wrapper i jednostka muszą używać tego samego ocenionego środowiska.
Nie zastępować ich lokalnym drop-inem. Zmiana interpretera wymaga nowego
zatwierdzenia profilu, a nie obejścia aktualnej kwalifikacji.
Materiały fazy4 ze scope v1 służą obecnie próbie odmowy. Przed fazą dodatnią
zaktualizować builder i tę instrukcję do wdrożonej, zatwierdzonej wersji
kontraktu wiążącego environment-id; nie dopisywać dowolnych pól do starego
manifestu i nie przedstawiać go jako certyfikatu nowego środowiska.

Sprawdzić osobno metadane `/etc/postgresql`, `/etc/postgresql/16`,
`/etc/postgresql/16/main` i `conf.d`. Obecny instalator wymaga root-owned0755;
standardowe postgres-owned katalogi mogą zostać odrzucone. Nie naprawiać ich
rekurencyjnie. Ewentualne przygotowanie pojedynczych ścieżek jest osobną
zatwierdzoną operacją administratora tej jednorazowej VM, po analizie wpływu
na zarządzanie pakietami i odzyskiwanie. Utrwalić każde stare/nowe metadata.
Każda niewyjaśniona odmowa pozostawia próbę BLOCKED.

Po spełnieniu tych warunków zapisać snapshot `S3-approved-prerequisites`.
Ponowić dokładny, rzeczywisty preflight nowego zatwierdzonego profilu;
konieczne są kod0, `READY`, stan `fresh` i właściwy hash manifestu. Następnie
wykonać to samo polecenie z `--install`, zachowując stdout i kod wyjścia.
Oczekiwane: kod0, `INSTALLED`, `WITHHELD`. Nie wywoływać automatycznie
daemon-reload/enable/start ani stosować nowych ustawień PostgreSQL.

Po instalacji odczytowo sprawdzić:

- dokładny inwentarz `/usr/lib/cryptohunter`, bez dodatkowych plików/katalogów,
  hash i rozmiar każdego pliku zgodny z podpisanym manifestem, tryby0644/0755,
  UID0, katalogi0755; brak symlinków, hardlinków, `.pth` i lokalnego cache;
- jednostkę root:root0644 i dwa pliki HBA/ident oraz plik w `conf.d`
  root:postgres0640, z dokładnymi hashami odpowiednich źródeł manifestu;
- katalog konfiguracji root:postgres0750 i niezmienione pozostałe dane;
- `/var/lib/cryptohunter-installer`0700, `journal.json` i `receipt.json`0600,
  oba z właściwym SHA źródła, hashem manifestu, zakresem i statusem `complete`;
- brak stagingu tego manifestu, brak startu/włączenia usługi, niezmieniony
  stan klastra i identyczny wynik zapytania do `installation_guard`.

Każdą rozbieżność zachować jako failure; nie przepisywać receipt. Ponownie
wykonać rzeczywisty preflight i identyczne `--install`. Mają zwrócić stan
`installed` i nie zmienić inwentarza, konfiguracji ani dziennika. Jest to
test idempotencji bajtów; nadal nie kwalifikuje działającego produktu.

## 7. Odmowy i przerwane instalacje — osobny snapshot na przypadek

Każdy przypadek zaczynać z właściwego snapshotu, bez automatycznej naprawy.
Zmienić tylko jeden warunek. Zachować stan przed/po, JSON i kod wyjścia.
Sprawdzić niezmienność chronionych ścieżek, backupu i danych guard.
Test nie jest zaliczony, jeśli wcześniejsza, niezależna bramka zasłoniła
zamierzoną odmowę. Na standardowym Noble kod20 zasłania próby NSS/klastra;
nie oznaczać ich jako wykonane.

| Przypadek | Przygotowanie wyłącznie przez operatora VM | Oczekiwany wynik |
| --- | --- | --- |
| Brak klastra | Powrót do S1; nie usuwać istniejącego klastra | Po spełnieniu wcześniejszych bramek kod21 i zero zmian |
| Konflikt konta/grupy | Osobny snapshot; przygotować nazwę z innym primary GID lub dodatkowego członka IPC | Kod22, bez naprawiania NSS |
| Niezgodne prawa | Zmienić jeden zatwierdzony katalog na writable lub niewłaściwy tryb | Kod23 i zero zmian |
| Artefakt podmieniony | W rootless kopii bundle zmienić bajt pliku, zostawiając podpisany manifest | Kod12; działa także przed bramką zależności |
| Manifest podmieniony | Zmienić bajt manifestu bez nowego podpisu | Kod11 |
| Symlink/ucieczka | Jeden plik bundle zastąpić linkiem; cel poza bundle ma osobny sentinel | Odmowa12/23, cel niezmieniony |
| Obcy istniejący kod | W osobnym snapshotcie utworzyć `/usr/lib/cryptohunter/keep.txt` z sentinel | Kod24, sentinel zachowany |
| Obca konfiguracja | Utworzyć jeden plik docelowy z inną treścią | Kod24, bajty zachowane |
| Drop-in lub vendor unit | W osobnej próbie przygotować niezatwierdzony fragment systemd | Kod24; nie uruchamiać podstawionej usługi |
| Częściowe wdrożenie | Pozostałości z przerwanego przebiegu, brak kompletnego receipt | Kod24, bez cleanup/adopcji |
| Inne wydanie | Inny prawidłowo podpisany inwentarz przeciw istniejącemu kompletnemu wdrożeniu | Odmowa24, stare dane i kod zachowane |

Próbę przerwania wykonać dopiero po otwarciu fazy dodatniej i z kopii
`S3-approved-prerequisites`. Operator w root shell tej VM może uruchomić
rzeczywisty wrapper pod `strace`, opóźniając pierwsze `renameat2` przez
`-e inject=renameat2:delay_enter=15s:when=1`. To jawna instrumentacja fault
testu, nie zwykła instalacja. Zachować logi i PID zarówno tracera, jak
instalatora; przed sygnałem sprawdzić UID0 i dokładne cmdline właściwego PID.
W drugim terminalu wysłać SIGTERM wyłącznie do tego procesu instalatora,
następnie poczekać na zakończenie i zapisać status. Nie używać `pkill -f`.

W terminalu1, po ponownym zatwierdzeniu bieżącego profilu i materiałów:

```sh
sudo strace -ff -e trace=renameat2 \
  -e inject=renameat2:delay_enter=15s:when=1 \
  -o "$trial_evidence/interrupt.trace" \
  /opt/approved-installer/deployment/install-production-local.sh \
  --install --bundle "$trial_release/payload" \
  --manifest "$trial_release/release.json" --signature "$trial_release/release.sig" \
  --trusted-key /etc/cryptohunter-vm-test-authority.pub \
  --expected-revision "$trial_revision" \
  > "$trial_evidence/interrupt.json"
trial_interrupt_rc=$?
printf '%s\n' "$trial_interrupt_rc" > "$trial_evidence/interrupt.exit"
```

Terminal2 służy wyłącznie ręcznemu wyborowi i sprawdzeniu właściwego PID
przez `ps -eo pid,ppid,euid,args` oraz `/proc/PID/cmdline`; następnie operator
wykonuje `sudo kill -TERM PID` dla sprawdzonego procesu. PID trzeba przepisać
z tej próby, nie z poprzedniego logu. Nie wysyłać sygnału do sudo, tracera,
PostgreSQL ani procesu usługi. Zachować dowód wybranego PID i cmdline.

Oczekiwany jest kod30 i zachowany dziennik/staging lub opublikowany kod,
zależnie od ostatniej wykonanej operacji. Rzeczywisty stan ma rozstrzygać,
nie założenie o momencie sygnału. Jeśli przebieg zakończył się wcześniej kodem0,
nie zaliczać przerwania: zachować wynik i powtórzyć z nowego snapshotu.
Osobna odtwarzalna odmowa publikacji może użyć
`-e inject=renameat2:error=EIO:when=1`, również wyłącznie w tej VM.
Nie dodawać produkcyjnego parametru obchodzącego preflight lub failpointu.

Po awarii ponowny `--preflight`/`--install` ma odmówić częściowego stanu,
nie dokończyć go automatycznie. Test zaniku zasilania wykonuje operator
przez hypervisor wyłącznie na osobnej kopii dysku; nie na Dellu i nie w CI.

## 8. Ręczne recovery i zamknięcie próby

Po awarii zatrzymać kolejne mutacje. Zachować dysk/snapshot, podpisany manifest,
JSON, kody, logi, backup danych, `journal.json`, `receipt.json`, jeżeli istnieje,
oraz `/usr/lib/.cryptohunter-stage-<hash_manifestu>`. Dziennik może być opóźniony
względem ostatniej utrwalonej operacji; nie jest automatycznym planem cofania.

Operator porównuje każdy opublikowany plik z podpisanym inwentarzem, stanem
przed próbą, własnością i trybem. Jawnie rozdziela: nowe kopie kodu/config
oraz wcześniejsze dane, konta/grupy, klaster i klucze. Nie usuwać nieznanych
plików, danych PG ani kont tylko dlatego, że ostatni wpis jest niekompletny.
Nie dopisywać ręcznie `complete` i nie kasować dziennika w celu ponowienia.

Wybrać i zatwierdzić jedną ścieżkę: zachowanie dysku do analizy i odtworzenie
**całej jednorazowej VM** z S3 do dalszych prób, albo osobny, dokładny plan
ręcznych operacji na potwierdzonych nowych plikach, z backupem i ponowną
weryfikacją wszystkich danych. Odzyskiwanie nie ma automatycznie usuwać
systemowych kont, klastra ani schema. Przywrócenie snapshotu jest uprawnieniem
tej izolowanej próby, a nie procedurą cofania produkcyjnego hosta.

Końcowy protokół musi podać obrazy/snapshoty, dokładny commit i environment-id,
materiały podpisane TEST ONLY, wersje pakietów, każdą wykonaną/niezakończoną
próbę, wyniki integralności/danych, odstępstwa oraz status recovery.
`BLOCKED` i `NOT EXECUTED` nie są PASS. Raport zakończyć bez Stage9 COMPLETE,
bez `LINUX_PRODUCTION_READY=PASS`, bez merge i bez kwalifikowania fizycznego Della.
