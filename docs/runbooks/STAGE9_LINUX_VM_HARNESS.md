# Stage 9 Linux — harness jednorazowej VM i kontrakt v2

Ta procedura dotyczy wyłącznie niezależnie przygotowanej i jawnie zatwierdzonej
Ubuntu Server 24.04 amd64 VM, z rzeczywistym systemd jako PID 1 i PostgreSQL 16.
Nie kwalifikuje fizycznego hosta ani produkcji i nie awansuje Stage 9.
Kontener, systemd-analyze bez działającego systemd oraz syntetyczna sonda SQL
nie zastępują próby VM. Przy braku takiej VM wynik natywny pozostaje `PENDING`.

**Bieżący kontrakt instalatora to `cryptohunter.freshness.substrate.v2`,
schema_version=2.** Historyczna procedura
[manifestu v1](STAGE9_LINUX_UBUNTU2404_VM_TRIAL.md) zachowuje kontekst wcześniejszej
odmowy i przygotowania VM; nie uruchamiać jej eksportu ani fazy dodatniej jako
próby aktualnego wydania. Obowiązuje poniższy eksport v2 oraz
[kontrakt podpisanego runtime](../architecture/stage9_linux_signed_python_runtime.md).

## 1. Podział odpowiedzialności i snapshoty

Operator hypervisora niezależnie weryfikuje oficjalny obraz Ubuntu, podpisane
sumy i SHA-256, architekturę amd64, wyłączenie shared folders/clipboard,
przekazywania urządzeń, dostępu do Della oraz materiałów produkcyjnych.
Materiały przenosi przez odłączany nośnik, nigdy współdzielony katalog hosta.
Po pobraniu oficjalnych pakietów i zatwierdzonego wheelhouse odłącza interfejs
sieciowy VM w hypervisorze. Dowód izolacji i obrazu zachowuje poza gościem.

Harness nie tworzy VM, kont OS, klastra PostgreSQL, kluczy ani autorytetów.
Nie wykonuje APT ani nie przywraca snapshotów. Operator przygotowuje je osobno
w pustej VM i zatrzymuje się przy zastanych kontach, klastrach i plikach.
Nie adoptować ani usuwać obcego wdrożenia. Używać kont i grup zgodnych
z `deployment/production_local_installer_host.py` i właściwym unit systemd.

Zachować co najmniej snapshoty:

| Snapshot | Zawartość i przeznaczenie |
| --- | --- |
| S0-clean | Zweryfikowany obraz przed przygotowaniem. |
| S1-packages | Oficjalne pakiety, zachowane źródła APT i dokładne wersje. |
| S2-substrate | Konta, grupa IPC, pojedynczy klaster 16/main online na 5432, syntetyczny guard danych. |
| S3-approved-prerequisites | Odłączona sieć, przejrzany bootstrap, zatwierdzone wejścia i approval. |
| S4-runtime | Zweryfikowane opublikowane wydanie i chroniony binding runtime. |
| S5-installed | Zainstalowana aplikacja, receipt, guard danych bez zmian. |

Przed snapshotem zachować spójne dowody; hash dysku wyliczać po wyłączeniu VM,
nie z aktywnie zmienianego obrazu. Każda próba negatywna zaczyna się od osobnego
klona odpowiedniego snapshotu. Po próbie zachować dowody i odtworzyć **całą VM**.
Nie usuwać ręcznie journala/stagingu, nie uruchamiać automatycznego recovery
ani rekursywnego chown/chmod zastanego drzewa.

Do przygotowania potrzebne są oficjalne pakiety `python3`, `python3-venv`,
`python3.12`, `python3.12-venv`, `postgresql-16`, `postgresql-client-16`,
`postgresql-common`, `systemd`, `dbus`, `openssl`, `binutils`, `iproute2` i `ca-certificates`.
Nie używać systemowych cryptography/psycopg jako zamiennika podpisanego runtime;
nie modyfikować systemowego Pythona ani używać `--break-system-packages`.
Zachować `/etc/os-release`, `uname`, PID 1, wersję CPython/systemd, APT sources,
`dpkg-query -W`, listę klastrów i konfigurację hypervisora.

Przygotowanie 16/main i kont opisują sekcje 2–3 historycznego runbooka, ale jego
listy bibliotek Pythona i manifestu v1 nie obowiązują nowego profilu.
Osobny syntetyczny `stage9_vm_guard` i jego sprawdzony backup służą porównaniu
wierszy przed/po instalacji; nie są schematem Freshness ani poświadczeniem.
Schemat Freshness przygotować tylko przez istniejące, niezależnie przejrzane
narzędzia. Nie tworzyć syntetycznych poświadczeń PRODUCTION, aby odblokować
dodatni scenariusz. Brak właściwych autorytetów ma pozostać `BLOCKED`.

## 2. INSPECT — wyłącznie odczyt

Na dowolnej maszynie można uruchomić bez sudo:

```sh
python -I -s -B deployment/stage9_vm_trial.py --mode INSPECT
```

Tryb wypisuje JSON na stdout. Nie tworzy raportu, katalogów, kont ani klastra,
nie instaluje i nie uruchamia usług. Odczytuje obserwacje środowiska; ich brak
nie daje uprawnienia do wykonania. Redirekcja stdout do pliku jest odrębną
decyzją operatora i sama jest zapisem, poza odczytowym kontraktem INSPECT.

## 3. Zatwierdzenie VM przed EXECUTE-VM-TRIAL

Zatwierdzenie wiąże konkretny boot, maszynę, commit, environment-id, obraz,
snapshot i dokładne wersje pakietów. Plik
`/etc/cryptohunter/stage9-vm-trial-approval.json` przygotowuje uprawniony operator
po niezależnej kontroli. Cały łańcuch ścieżki musi być root-owned, bez symlinków
i zapisu grupy/innych; plik zwykły, pojedynczy hardlink, maksymalnie 64 KiB.
Nie generować zgody automatycznie na podstawie samej inspekcji.

Dokładny kontrakt JSON, bez dodatkowych lub powtórzonych pól:

```json
{
  "schema": 1,
  "purpose": "TEST_ONLY",
  "approved": true,
  "disposable_vm": true,
  "isolation_no_host_shares": true,
  "network_disconnected": true,
  "no_production_material": true,
  "machine_id": "<32 lowerhex z /etc/machine-id>",
  "boot_id": "<UUID z /proc/sys/kernel/random/boot_id>",
  "revision": "<zatwierdzone 40 lowerhex>",
  "environment_id": "noble-cp312-trial-001",
  "vm_image_sha256": "<64 lowerhex zweryfikowanego obrazu>",
  "vm_snapshot": "S3-approved-prerequisites",
  "packages": {
    "systemd": "<dokładna wersja dpkg>",
    "python3.12": "<dokładna wersja dpkg>",
    "python3.12-venv": "<dokładna wersja dpkg>",
    "postgresql-16": "<dokładna wersja dpkg>",
    "postgresql-client-16": "<dokładna wersja dpkg>",
    "postgresql-common": "<dokładna wersja dpkg>",
    "openssl": "<dokładna wersja dpkg>"
  }
}
```

Znaczniki zastąpić rzeczywistymi zatwierdzonymi wartościami. Wartości `true`
są deklaracjami odpowiedzialnego operatora, nie atestacją hypervisora.
Guard wymaga zgodnych obserwacji Ubuntu 24.04/amd64, systemd PID 1,
wirtualizacji, braku kontenera, sieci i udziałów oraz wersji pakietów.
Niezgodność zatrzymuje wykonanie przed czynnością uprzywilejowaną.
Po reboot/restore zmieniającym boot-id approval wymaga ponownej niezależnej
oceny. Zmiana systemowego interpretera lub bibliotek wymaga ponownego przeglądu
i podpisu runtime, nie automatycznej aktualizacji hashy.

Uruchamiany jako root kod harnessu, jego moduły i bootstrap muszą pochodzić
z przejrzanego commitu w chronionej kopii `/srv/cryptohunter-trusted`.
Nie uruchamiać jako root edytowalnego checkoutu. Operator osobno weryfikuje
podpisany wykaz hashy tej kopii; approval nie podpisuje kodu.

## 4. Runtime, niezależny podpis i binding

Stosować istniejący builder i pełny lock/katalog wheelhouse, bez zastępowania
hashy plikami znalezionymi lokalnie. Pobieranie odbywa się przed izolacją:

```sh
/usr/bin/python3.12 -I -s -B deployment/build_freshness_runtime.py \
  --download-wheelhouse /bezwzgledna/sciezka/wheelhouse
```

W zgodnej Ubuntu 24.04 VM, jako nie-root i już offline:

```sh
trial_revision=$(git rev-parse HEAD)
/usr/bin/python3.12 -I -s -B deployment/build_freshness_runtime.py \
  --wheelhouse /bezwzgledna/sciezka/wheelhouse \
  --destination /bezwzgledna/sciezka/kandydat-runtime \
  --environment-id noble-cp312-trial-001 \
  --release-revision "$trial_revision" \
  --manifest /bezwzgledna/sciezka/runtime-manifest.json
```

Destination i manifest muszą być nowe. Zachować `build-probe.json`, lock,
katalog wheels, wszystkie hashe, unsigned manifest, inwentarz stdlib/native
i faktyczne wyniki importów cryptography/psycopg, Ed25519, OpenSSL i libpq.
Wymagane CPython 3.12, ABI `cpython-312-x86_64-linux-gnu`, bez system-site,
user-site, `.pth` ani dodatkowych pakietów. Sonda nie dowodzi połączenia z bazą.

Niezależna rola podpisuje dokładne bajty manifestu nowym autorytetem Ed25519
TEST_ONLY na odrębnym stanowisku. Ceremonia i raw32 są w
[procedurze podpisanego runtime](STAGE9_SIGNED_RUNTIME_VM_TRIAL.md#3-niezależny-podpis-test-only).
Prywatny klucz nie trafia do VM wykonawczej, bundle, runtime ani logów.
Publiczny raw32 dostarczyć oddzielnym kanałem; nie ufać automatycznie kluczowi
dołączonemu do materiałów. Autorytet aplikacji ma być odrębny od runtime.

Podpisany manifest wskazuje `installation_path`. Niezależnie przejrzany binding
`/etc/cryptohunter/freshness-runtime.json` ma root:root, `0644`, chronionych
przodków i dokładnie następujące pola:

```json
{
  "schema": 1,
  "environment_id": "noble-cp312-trial-001",
  "manifest_sha256": "<hash dokładnych podpisanych bajtów runtime-manifest.json>",
  "source_revision": "<zatwierdzone 40 hex>",
  "executable": "/opt/cryptohunter/freshness-runtime/releases/noble-cp312-trial-001/bin/python3.12",
  "base_executable": "/usr/bin/python3.12",
  "base_executable_sha256": "<sha256 wpisu external_files /usr/bin/python3.12>"
}
```

Wartości odtworzyć z wcześniej niezależnie zweryfikowanego manifestu, bez
sondowania obcego runtime i bez przeliczania zatwierdzonych hashy z hosta.
Przygotowanie chronionego bindingu i katalogu bazowego jest osobną czynnością
operatora; publikacja runtime nie jest instalacją aplikacji.

## 5. Eksport aplikacji v2 z zatwierdzonego commitu

Na stanowisku przygotowania, bez root, użyć wersjonowanego eksportera.
Najpierw zatwierdzić dokładny commit i wcześniej niezależnie zweryfikowany
binding z sekcji 4. Katalog wyjściowy musi być nowy, a checkout czysty i zgodny
z tym SHA:

```sh
trial_revision=$(git rev-parse HEAD)
python -I -s -B scripts/prepare_stage9_vm_bundle.py \
  --source /bezwzgledna/sciezka/checkout \
  --revision "$trial_revision" \
  --binding /bezwzgledna/sciezka/runtime-binding.json \
  --output /bezwzgledna/sciezka/stage9-vm-release-v2
```

Eksporter czyta `SCOPE`, `REQUIRED_FILES` i kanoniczny `UNIT_TEMPLATE` przez AST,
bez importowania kodu produktu. Sprawdza Git blobs i zgodność lokalnych bajtów
z zatwierdzonym SHA; odmawia symlinków, obcego checkoutu i zastanego outputu.
Aktualna baza ma 15 plików, lecz źródłowy zbiór jest kontraktem, nie historyczna
liczba. Unit materializuje dokładną ścieżką executable z bindingu **przed**
hashowaniem. Wynik zawiera `payload/`, unsigned `release.json` v2 z obiektem
runtime równym bindingowi bez `schema`, pięć plików bootstrapu instalatora
w `bootstrap/deployment/`, `bootstrap.sha256` i `revision.txt`.

Oddzielnie przygotować i podpisać wykaz kodu harnessu i jego zależności
(`stage9_vm_trial.py`, `stage9_vm_environment.py`, `stage9_vm_native.py` oraz
moduły importowane przez natywne obserwacje) z tego samego SHA. Nie należą one
do minimalnego payloadu usługi. Najprościej zachować pełny przejrzany eksport
`bot_core/` i `deployment/` w chronionej kopii `/srv/cryptohunter-trusted`;
nie używać niezweryfikowanych sibling modules z innego checkoutu.

Odrębna rola porównuje binding, szablon unit i wszystkie eksportowane bajty
z zatwierdzonym SHA przed podpisaniem. Nie dodawać pól v2 do manifestu v1,
nie przepisywać JSON po podpisie i nie uznawać unsigned eksportu za zatwierdzenie.

Na odrębnym stanowisku nowy autorytet aplikacji TEST_ONLY podpisuje aplikację
i wykaz bootstrapu. Prywatny klucz pozostaje na tym stanowisku:

```sh
umask 077
openssl genpkey -algorithm ED25519 -out app-test-only.pem
openssl pkey -in app-test-only.pem -pubout -out app-test-only.pub.pem
openssl pkey -in app-test-only.pem -pubout -outform DER -out app-test-only.der
python -I -s -B - <<'PY'
from pathlib import Path
raw = Path('app-test-only.der').read_bytes()
assert len(raw) == 44 and raw[:12].hex() == '302a300506032b6570032100'
Path('app-test-only.raw').write_bytes(raw[12:])
PY
openssl pkeyutl -sign -rawin -inkey app-test-only.pem \
  -in stage9-vm-release-v2/release.json -out stage9-vm-release-v2/release.sig
openssl pkeyutl -sign -rawin -inkey app-test-only.pem \
  -in stage9-vm-release-v2/bootstrap.sha256 -out stage9-vm-release-v2/bootstrap.sig
```

Operator VM porównuje fingerprint raw32 oddzielnym kanałem, sprawdza oba podpisy
zaufanym systemowym OpenSSL i `sha256sum -c` przed instalacją chronionej kopii
bootstrapu. Nie importować kodu z niezweryfikowanego bundle jako root.
Nie kopiować `app-test-only.pem` do nośnika ani VM.

## 6. Wejścia i kolejne fazy wykonania

Przygotować nowy chroniony katalog wejść, np. `/srv/stage9-vm-input/trial-001`,
z root-owned łańcuchem przodków, bez zapisu grupy/innych. Zachować literalne
zatwierdzone symlinki wewnątrz runtime; nie dereferencjonować ich przy kopiowaniu.
Nie akceptować symlinków zamiast katalogu bundle, manifestu, podpisu i klucza.
Publiczny klucz runtime instalować oddzielnie pod
`/etc/cryptohunter/freshness-runtime-authority.pub`; klucz aplikacji pod
`/etc/cryptohunter-vm-test-authority.pub`. Klucze muszą być niezależnie zaufane
raw32 TEST_ONLY; sama własność root nie ustanawia autorytetu.

`inputs.json` zawiera wyłącznie te dwa obiekty i cztery ścieżki w każdym:

```json
{
  "runtime": {
    "runtime": "/srv/stage9-vm-input/trial-001/runtime",
    "manifest": "/srv/stage9-vm-input/trial-001/runtime-manifest.json",
    "signature": "/srv/stage9-vm-input/trial-001/runtime-manifest.sig",
    "trusted_key": "/etc/cryptohunter/freshness-runtime-authority.pub"
  },
  "application": {
    "bundle": "/srv/stage9-vm-input/trial-001/payload",
    "manifest": "/srv/stage9-vm-input/trial-001/release.json",
    "signature": "/srv/stage9-vm-input/trial-001/release.sig",
    "trusted_key": "/etc/cryptohunter-vm-test-authority.pub"
  }
}
```

Po zatwierdzeniu VM operator wykonuje pojedynczą fazę, zachowując stdout i kod
wyjścia w nowym katalogu dowodów poza wejściami. Używa systemowego interpretera
i chronionej kopii harnessu:

```sh
sudo /usr/bin/python3.12 -I -s -B \
  /srv/cryptohunter-trusted/deployment/stage9_vm_trial.py \
  --mode EXECUTE-VM-TRIAL \
  --approval /etc/cryptohunter/stage9-vm-trial-approval.json \
  --revision "$trial_revision" --environment-id noble-cp312-trial-001 \
  --phase runtime-verify --inputs /srv/stage9-vm-input/trial-001/inputs.json
```

Po sprawdzeniu wyniku kolejno stosować `--phase runtime-publish`, następnie
`--phase preflight` i dopiero po jego pozytywnym wyniku `--phase install`.
Każde uruchomienie ponownie wymaga guard VM; nie wyciągać zgody z wcześniejszego
PASS. Zatrzymać ciąg przy odmowie, braku danych albo niezgodności.

Preflight jest odczytowy. Harness porównuje stan chronionych ścieżek przed/po;
operator zachowuje dodatkowo NSS, listę klastrów, stan usług, wiersze guard
i hash backupu. Nie porównywać atime, zmienianego samym odczytem. Nowy journal,
staging, plik lub zmiana danych oznacza `FAIL`, nawet jeżeli instalator zwrócił
READY. Porównanie stanu działa również przy wyjątku lub odmowie sprawdzanej
operacji, a wykryta mutacja ma status `FAIL`, nie `BLOCKED`. Instalacja nie
tworzy kont, klastra, schematu ani nie restartuje usług.
Po instalacji porównać receipt, pliki, metadane, kompletność publikacji, journal
i guard danych. Powtórzyć identyczną instalację na osobnym klonie S5 oraz
odmowę obcego wydania na innym klonie; nie zaliczać przypadkowej odmowy podpisu
jako dowodu ochrony przed overwrite.

Fazy `systemd` i `postgresql-ipc` uruchamia się tym samym poleceniem operatora
z systemowym CPython. Harness najpierw wykonuje systemowym bootstrapem
`freshness_runtime_bootstrap.py verify-installed`, sprawdza environment-id
i rewizję w chronionym bindingu, a dopiero potem uruchamia tę samą chronioną
CLI przez zweryfikowany podpisany interpreter z `-I -s -B`. Wewnętrzny worker
ponawia pełny guard VM, walidację wejść i preflight oraz sprawdza zgodność
własnego `sys.executable` z bindingiem. Pozwala to użyć zatwierdzonego psycopg
w walidacji IPC bez instalowania pakietów ani modyfikowania systemowego Pythona.
Operator nie wybiera alternatywnego interpretera ani nie pomija bootstrapu.

Faza `systemd` dotyczy właściwego zainstalowanego unit: najpierw natywne
`systemd-analyze verify`, następnie rzeczywisty start i istniejący qualifier
`qualify_freshness_service_identity`. Zachować także końcowy stan usługi,
MainPID/ExecMainPID, ExecStartPre/ExecStart, UID/GID, cgroup, `/proc/PID/exe`,
PIDFD, inode/hash i sandbox. Nie zastępować service ani PostgreSQL unit atrapą.
Brak zależności musi mieć konkretny kod/przyczynę i status `BLOCKED` lub `FAIL`.

Przed fazą `postgresql-ipc` operator osobno przygotowuje zatwierdzony
`/etc/cryptohunter/linux-qualification.json`, root:root `0644`, bez symlinków,
hardlinków i dodatkowych ACL: `{"schema_version":1,"installation_id":"<64 hex>"}`.
Installation-id jest niezależnym identyfikatorem, nie samoodwołującym hashem
manifestu. Obecny instalator nie publikuje tego pliku. Brak polityki oznacza
brak diagnostycznej gałęzi root, nie udaną próbę IPC.

Istniejący `collect_verifier_observation` przypina socket i proces verifiera
oraz obserwuje połączenie PostgreSQL otwarte przez **ten proces**: peer/HBA,
backend PID, role, lokalny transport, cztery odmowy `42501`, rollback i stan
połączenia. Sonda administracyjna lub osobny proces z UID usługi nie dowodzi
własnego połączenia tej usługi. Ta częściowa obserwacja zachowuje
`production_qualification=BLOCKED` i wszystkie blockery runtime/replay.

## 7. Macierz negatywna i granice odmowy

Każdy wiersz ma początkowo `NOT_EXECUTED`. Zmiana statusu wymaga identyfikatora
osobnego snapshotu, pojedynczej opisanej mutacji, zdrowej kontroli bazowej,
wywołania, kodu odmowy, stanu przed/po i dowodu odmowy na właściwej granicy.
Nie wystarcza dowolny niezerowy exit. Harness nie wstrzykuje automatycznie
awarii ani nie usuwa skutków negatywnych prób.

| Osobny przypadek | Oczekiwana granica | Stan początkowy |
| --- | --- | --- |
| Zmieniony manifest runtime / błędny podpis | Niezależny Ed25519 przed interpretacją i wykonaniem runtime | NOT_EXECUTED |
| Zmieniony plik runtime / hash / tryb | Inwentarz podpisanego runtime | NOT_EXECUTED |
| Brakujący lub dodatkowy wheel/pakiet, `.pth`, system-site | Builder/inwentarz/import isolation | NOT_EXECUTED |
| Inny ABI, environment-id lub rewizja | Podpisany kontrakt i binding | NOT_EXECUTED |
| Podmieniony interpreter lub obcy symlink | Podpisany literalny link, external_files i program procesu | NOT_EXECUTED |
| Zmieniony manifest/podpis aplikacji | Autorytet aplikacji i ścisły manifest v2 | NOT_EXECUTED |
| Niezatwierdzony drop-in lub zmieniony ExecStart/ExecStartPre | Efektywna konfiguracja systemd i D-Bus | NOT_EXECUTED |
| Inny UID/GID/grupy lub uprawnienia socketu/katalogu | Tożsamość procesu i IPC | NOT_EXECUTED |
| Podmiana procesu, inode/hash, kończący się peer, PID reuse | PIDFD, lifetime i powtórna obserwacja | NOT_EXECUTED |
| Zmieniony klaster, HBA/ident, endpoint lub autorytet | Właściwe połączenie PG i niezależna konfiguracja | NOT_EXECUTED |
| Utrata połączenia / niedostępna baza | Connect-before-bind lub odmowa UNAVAILABLE; brak fałszywego READY | NOT_EXECUTED |
| Przerwana publikacja przed rename / po rename | Journal i publikacja no-overwrite | NOT_EXECUTED |
| Pozostawiony staging/journal | Preflight odmawia bez automatycznego cleanup | NOT_EXECUTED |
| Próba nadpisania obcego wydania / danych | No-replace i brak zmian guard | NOT_EXECUTED |
| Powtórzenie identycznego wydania | Idempotencja oraz pełne ponowne sprawdzenie | NOT_EXECUTED |

### Odtworzenie pojedynczych mutacji

Poniższych poleceń **nie wykonano w tej sesji**. Każdy przypadek wykonuje
operator na osobnym klonie zatwierdzonej VM, po odtworzeniu zdrowego snapshotu
i ponownej ocenie approval dla bieżącego boot-id. Najpierw uruchamia dokładną
fazę kontrolną poleceniem z sekcji 6 i zachowuje jej `checks.<faza>.status=PASS`,
kod wyjścia, hashe wejść i stan przed. Brak zdrowej kontroli bazowej blokuje
test mutacji. Nie wykonywać wszystkich wariantów w jednym klonie.

Przykładowa pojedyncza mutacja wejścia runtime; ostatni argument to **jeden**
z `manifest`, `signature`, `runtime-file`, `extra-package`, `missing-wheel`,
`symlink`. Poniżej wybrano wyłącznie `manifest`:

```sh
sudo /usr/bin/python3.12 -I -s -B - manifest <<'PY'
import os
import stat
import sys
from pathlib import Path

root = Path('/srv/stage9-vm-input/trial-001')
runtime = root / 'runtime'
case = sys.argv[1]
if case in {'manifest', 'signature'}:
    target = root / ('runtime-manifest.json' if case == 'manifest'
                     else 'runtime-manifest.sig')
    assert stat.S_ISREG(target.lstat().st_mode) and target.stat().st_nlink == 1
    with target.open('r+b') as stream:
        first = stream.read(1)
        assert len(first) == 1
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
        stream.flush()
        os.fsync(stream.fileno())
elif case == 'runtime-file':
    target = runtime / 'requirements.lock'
    assert stat.S_ISREG(target.lstat().st_mode)
    with target.open('ab') as stream:
        stream.write(b'\n# TEST_ONLY: single mutation\n')
elif case == 'extra-package':
    target = runtime / 'lib/python3.12/site-packages/stage9_unapproved_module.py'
    with target.open('xb') as stream:
        stream.write(b'# TEST_ONLY: never execute this unapproved module\n')
    target.chmod(0o644)
elif case == 'missing-wheel':
    wheels = sorted((runtime / 'wheelhouse').glob('*.whl'))
    assert wheels and not (root / 'withheld-wheel').exists()
    wheels[0].rename(root / 'withheld-wheel')
elif case == 'symlink':
    target = runtime / 'bin/python3.12'
    assert target.is_symlink() and os.readlink(target) == '/usr/bin/python3.12'
    target.unlink()
    target.symlink_to('/usr/bin/python3')
else:
    raise SystemExit('Nieznany pojedynczy przypadek')
PY
```

Nie podpisywać ponownie zmienionych materiałów ani aktualizować manifestu.
Powtórzyć `--phase runtime-verify`, zachować pełny JSON i kod wyjścia. Zmiana
manifestu/podpisu powinna zostać odrzucona przez autorytet Ed25519 **przed**
interpretacją manifestu; zmiany drzewa przez podpisany inwentarz przed
wykonaniem interpretera. Usunięty wheel w tym przykładzie bada inwentarz
kandydata runtime. Osobny brak wheel **przed buildem** bada builder:
na osobnym klonie przenieść dokładnie jeden wheel z rootless wheelhouse poza
ten katalog i ponowić polecenie buildera do nowego destination; oczekiwana
jest odmowa `Niekompletny wheelhouse lub dodatkowy pakiet` przed tworzeniem
runtime. Nie zaliczać jej zamiast kontroli podpisu ani instalatora.

Harness celowo zwraca bezpieczny ogólny `SIGNED_RUNTIME_REFUSED`. Sama ta
przyczyna lub dowolny niezerowy exit nie dowodzą konkretnej granicy. Dla
powyższych przypadków można po pozytywnym guard VM zebrać wąską, odczytową
diagnozę z tego samego przejrzanego API. Ta diagnoza wypisuje jedynie
zatwierdzone stałe przyczyny, nigdy treści wejść:

```sh
sudo /usr/bin/python3.12 -I -s -B - "$trial_revision" noble-cp312-trial-001 <<'PY'
import runpy
import sys
from pathlib import Path

sys.pycache_prefix = '/dev/null'
trusted = Path('/srv/cryptohunter-trusted/deployment')
guard = runpy.run_path(str(trusted / 'stage9_vm_environment.py'))
guard['require_approved_vm'](
    Path('/etc/cryptohunter/stage9-vm-trial-approval.json'),
    revision=sys.argv[1], environment_id=sys.argv[2])
api = runpy.run_path(str(trusted / 'freshness_runtime.py'))
root = Path('/srv/stage9-vm-input/trial-001')
config = api['RuntimeConfig'](
    runtime=root / 'runtime', manifest=root / 'runtime-manifest.json',
    signature=root / 'runtime-manifest.sig',
    trusted_key=Path('/etc/cryptohunter/freshness-runtime-authority.pub'),
    expected_environment_id=sys.argv[2], expected_revision=sys.argv[1])
allowed = {
    'Nieprawidłowy podpis niezależnego autorytetu': 'ED25519_REFUSED',
    'Niepełny lub dodatkowy plik runtime': 'SIGNED_INVENTORY_REFUSED',
    'Niezgodna integralność pliku runtime': 'SIGNED_FILE_REFUSED',
    'Niezgodny symlink runtime': 'SIGNED_SYMLINK_REFUSED',
}
try:
    api['verify_runtime'](config)
except api['RuntimeTrustError'] as error:
    print(allowed.get(str(error), 'OTHER_BOUNDARY_UNPROVEN'))
    raise SystemExit(1)
except Exception:
    print('BOUNDARY_UNPROVEN')
    raise SystemExit(2)
print('UNEXPECTED_ACCEPTANCE')
raise SystemExit(3)
PY
```

Wcześniejsza odmowa approval, uprawnień albo brak OpenSSL nie zalicza testu
podpisu. `OTHER_BOUNDARY_UNPROVEN`/`BOUNDARY_UNPROVEN` wymaga wyjaśnienia;
pozostawić ocenę granicy `BLOCKED`. Po dowodach przywrócić całą VM.

Drop-in badać na osobnym klonie S5 ze zdrową bazową kontrolą instalacji.
Operator tworzy wyłącznie nowy, wcześniej nieistniejący katalog:

```sh
sudo /usr/bin/python3.12 -I -s -B - <<'PY'
from pathlib import Path
directory = Path('/etc/systemd/system/cryptohunter-freshness-verifier.service.d')
directory.mkdir(mode=0o755)
path = directory / '99-test-only.conf'
with path.open('x', encoding='ascii') as stream:
    stream.write('[Service]\nNoNewPrivileges=no\n')
path.chmod(0o644)
PY
```

Ponowić `--phase preflight`: oczekiwana jest odmowa niezatwierdzonego drop-in
bez daemon-reload ani startu usługi. Odmowa na tej granicy instalatora nie
zalicza osobnej kontroli **efektywnego** D-Bus. Jej ręczna próba wymaga własnego
zatwierdzonego klona, obserwacji managera i rzeczywistego kwalifikatora zgodnie
z mapowaniem źródeł poniżej. Nie uruchamiać osłabionej usługi dla samego testu.

Raport fazy odrzuconej mutacji pozostaje `FAIL` jako wynik wykonania harnessu.
Osobny ręczny testcase odmowy może otrzymać `PASS` dopiero po udokumentowaniu
zdrowej kontroli, jednej mutacji, dokładnej oczekiwanej granicy i zachowania
stanu. Nie przepisywać automatycznie `FAIL`/`BLOCKED` na PASS w JSON harnessu.

### Ręczne scenariusze hosta, procesu, PostgreSQL i recovery

Poniższe mapowanie wskazuje istniejący oracle i sposób przygotowania próby;
testy modeli nie są automatyzacją mutacji na VM. Wszystkie przypadki VM
pozostają `NOT_EXECUTED`, dopóki operator nie zachowa rzeczywistych dowodów.

| Przypadek i czynność na osobnym klonie | Istniejący oracle źródłowy |
| --- | --- |
| Zmiana UID/GID, grupy lub trybu jednego chronionego katalogu; preflight i porównanie stanu | `test_production_local_installer.py::test_system_group_collisions_and_ipc_membership_refuse`, `::test_group_writable_protected_parent_refuses_without_repair`; żywy proces: `test_freshness_service_identity.py::test_kernel_credentials_or_cgroup_drift_refuse` |
| Zmienione argv/ExecStart lub drop-in efektywnej jednostki; odczyt przez rzeczywisty qualifier, bez uruchamiania obcego polecenia | `test_freshness_service_identity.py::test_effective_configuration_drift_refuses_before_process_lookup`, `::test_alternative_binary_embedded_path_or_changed_arguments_are_refused` |
| Zakończenie przypiętego procesu lub podmiana obrazu na testowym klonie; utrzymać PIDFD i porównać okres życia/inode/hash | `test_freshness_service_identity.py::test_native_pidfd_is_bound_to_live_child_and_refuses_after_exit`, `::test_native_executable_inode_replacement_refuses_before_or_after_read`; IPC: `test_linux_service_ipc.py::test_dead_origin_is_refused_even_while_connected_socket_is_open` |
| Błędny ident/HBA/rola lub nadmierny grant; mutować wyłącznie tymczasowy klaster fixture | `test_linux_service_postgresql_integration.py::test_real_wrong_ident_mapping_refuses_connection`, `::test_real_trust_authentication_is_not_peer`, `::test_real_accidentally_granted_foreign_select_refuses` |
| Zerwanie backendu, niedostępna baza, restart jednorazowego klastra | `test_linux_service_postgresql_integration.py::test_real_backend_connection_loss_is_not_denial`, `::test_real_isolated_database_restart_reconnects_and_requalifies`; start usługi ma zachować connect-before-bind |
| Podmieniony endpoint lub partner procesu | `test_linux_verifier_observation.py::test_validator_refuses_raw_identity_challenge_application_and_pass_substitution`, `test_linux_service_ipc.py::test_pinned_real_peer_refuses_modelled_pid_reuse_program_or_cgroup_change` |
| Przerwanie publikacji przed/po rename; operator zatrzymuje proces tylko w osobnym klonie i zachowuje journal, staging, stan filesystemów | `test_freshness_runtime.py::test_partial_publication_refuses_retry`, `::test_crash_after_rename_blocks_installed_binding`; `test_production_local_installer.py::test_interruption_retains_recovery_evidence_and_refuses_automatic_retry` |
| Pozostały staging/journal; ponowienie preflight bez cleanup | `test_production_local_installer.py::test_extra_recovery_state_file_prevents_idempotent_adoption` i powyższe testy przerwania |
| Obce wdrożenie lub częściowa konfiguracja; porównać hashe i guard danych po odmowie | `test_production_local_installer.py::test_existing_unmanaged_installation_is_preserved_and_not_adopted`, `::test_partial_existing_configuration_refuses_without_overwriting` |

Nazwy plików w tabeli odnoszą się do `tests/deployment/` albo
`tests/security/` zgodnie z ich położeniem w repozytorium. Nie zabijać
przypadkowego procesu na podstawie samego PID ani nie zmieniać aktywnego
klastra instalacji, aby wykonać scenariusz fixture. W recovery nie ma gwarancji
trafienia w granicę rename samym czasem `kill`: bez dowodu konkretnego momentu
przerwania wynik tego przypadku pozostaje `BLOCKED`, nie PASS.

### Istniejące testy rzeczywistego PostgreSQL — osobne opt-in

Przed izolacją VM przygotować oddzielne przejrzane środowisko narzędzi pytest
`/srv/stage9-test-tools`, z zamrożonymi zależnościami, oraz kopię testów
`/srv/cryptohunter-reviewed-tests` z tego samego SHA. Nie dodawać pytest
ani zależności testowych do podpisanego runtime usługi.

Na zatwierdzonym klonie, z konta **nie-root**, istniejący adapterowy harness
tworzy własny tymczasowy klaster i używa wyłącznie fixture. Nie jest to proces
właściwej usługi ani dowód jej produkcyjnej separacji:

```sh
cd /srv/cryptohunter-reviewed-tests
CRYPTOHUNTER_RUN_LINUX_PG_ADAPTER_INTEGRATION=1 \
CRYPTOHUNTER_TEST_PG_BINDIR=/usr/lib/postgresql/16/bin \
  /srv/stage9-test-tools/bin/python -m pytest -q \
  tests/security/test_linux_service_postgresql_integration.py
```

Osobny root peer-auth fixture tworzy konta testowe i klaster. Uruchamiać go
wyłącznie na **odrębnym zatwierdzonym klonie**, po sprawdzeniu całej fixture,
z chronioną kopią kodu i narzędzi. Poniższy moduł tworzy izolowany schemat
sondy ACL, bez poświadczeń kryptograficznych produkcji:

```sh
cd /srv/cryptohunter-reviewed-tests
sudo env PATH=/usr/lib/postgresql/16/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /srv/stage9-test-tools/bin/python -m pytest -q \
  tests/security/test_postgresql_freshness_production_local_authentication.py
```

Nie uruchamiać z rozpędu wszystkich testów jako root ani nie tworzyć
syntetycznych poświadczeń PRODUCTION w instalacji VM. Dodatnie testy pełnej
semantyki w `test_freshness_semantic_verifier_postgresql.py` mają osobny zakres
fixture i nie otwierają bramek produktu; wymagają osobnej oceny autorytetów
TEST_ONLY. Po każdym uprzywilejowanym fixture zachować wynik i odtworzyć klona.
Testy native IPC mają własny `CRYPTOHUNTER_DISPOSABLE_ROOT_IPC=1`; nie ustawiają
go zwykłe workflow. Żaden powyższy wynik nie zamyka macierzy właściwej usługi.

## 8. Raport, znaczenie wyników i zatrzymanie

Raport zachowuje identyfikator i SHA obrazu, snapshoty, wersje systemu/pakietów,
zatwierdzony commit, environment-id, SHA-256 dokładnych manifestów, wyniki
podpisów, preflight wraz z porównaniem stanu, instalację, systemd, PostgreSQL,
IPC i osobne przypadki macierzy. Nie zapisywać kluczy prywatnych, poświadczeń,
surowego środowiska procesu ani danych poufnych. Dowody stdout/kody wyjścia
przechowywać razem z decyzją operatora i faktami izolacji spoza VM.

- `PASS`: wykonana dokładna kontrola i zachowane jej wymagane dowody.
- `FAIL`: wykonana kontrola naruszyła kontrakt lub otrzymano nieoczekiwany wynik.
- `BLOCKED`: konkretna brakująca zależność/warunek uniemożliwia kontrolę.
- `NOT_EXECUTED`: kontrola nie została wykonana.

Brak próby VM jest `PENDING` w końcowych flagach natywnych, a poszczególne
kontrole pozostają `NOT_EXECUTED`/`BLOCKED`. Podsumowanie nie może zamieniać
`BLOCKED`, pominięcia ani częściowej diagnostyki w `PASS`. Nie stosować
`all([])` jako dowodu powodzenia. Znane problemy reviewed source pins i brak
pełnego runtime są opisane w audycie; nie aktualizować hashy, by uzyskać zielony
wynik. Zakończyć bez merge, enrollment, TPM i awansowania Stage 9.
