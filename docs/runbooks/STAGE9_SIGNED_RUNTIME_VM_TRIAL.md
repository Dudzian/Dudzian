# Stage 9 — ręczna próba podpisanego runtime na jednorazowej VM

> Bieżące wykonanie prowadzić przez
> [harness INSPECT/EXECUTE-VM-TRIAL](STAGE9_LINUX_VM_HARNESS.md), z niezależnym
> zatwierdzeniem VM i manifestem aplikacji v2. Odsyłacze poniżej do starszego
> runbooka służą wyłącznie przygotowaniu/izolacji; jego eksport manifestu v1
> i historyczna blokada runtime nie są procedurą bieżącego wydania.

Procedura dotyczy wyłącznie Ubuntu Server 24.04 amd64 z systemd jako PID 1.
Wykonuje ją operator na odizolowanej VM, bez Della, danych lub kluczy
produkcyjnych. Nie wykonuje jej automatycznie pytest ani agent przygotowujący PR.
Wynik próby nie awansuje Stage 9. Kontener sprawdza build/importy, lecz nie
zastępuje tej próby instalatora, sandboxu i procesu systemd.

## 1. Przygotowanie i zachowanie wejść

Stosować izolację VM, weryfikację obrazu Ubuntu, snapshoty, zapis APT i zasadę
zatrzymania przy zastanych kontach/klastrach z
[dotychczasowej procedury instalatora](STAGE9_LINUX_UBUNTU2404_VM_TRIAL.md).
Jej historyczna blokada zależności dotyczy profilu sprzed tego runtime;
nowy profil wymaga opisanej tutaj osobnej ceremonii. Nie używać systemowych
pakietów cryptography/psycopg jako zamiennika.

Przed mutacjami operator archiwizuje `/etc/os-release`, wersję systemd,
`/usr/bin/python3.12 --version`, wyniki `dpkg-query -W` dla Python/OpenSSL/libc,
źródła APT oraz identyfikator snapshotu. Wymagane są oficjalne pakiety
Python 3.12, `python3.12-venv` i OpenSSL. PostgreSQL 16, odrębne konta i syntetyczne
dane przygotowuje operator według istniejącego runbooka, dopiero po własnej
ocenie. Agent nie tworzy kont, nie tworzy ani nie usuwa klastra.

Przeglądany checkout musi odpowiadać konkretnemu 40-znakowemu SHA. Wyeksportować
bootstrap/instalator do chronionej kopii `/srv/cryptohunter-trusted`:
root jest właścicielem całego łańcucha katalogów; brak symlinków i zapisu
przez grupę/innych. Nie uruchamiać jako root nieprzejrzanego checkoutu.

## 2. Wheelhouse i budowanie bez sieci

Na odrębnym stanowisku pobrać wyłącznie zamrożony katalog:

```sh
/usr/bin/python3.12 -I -s -B deployment/build_freshness_runtime.py \
  --download-wheelhouse /bezwzgledna/sciezka/wheelhouse
```

Zachować katalog JSON, lock, SHA-256 każdego wheel oraz ich źródła. Przenieść
materiały do VM przez odłączany nośnik. Po przygotowaniu APT odłączyć sieć VM.
Nie montować folderów hosta. Build wykonuje konto bez uprawnień administratora
z systemowym CPython 3.12, na zgodnym obrazie Ubuntu:

```sh
trial_revision=$(git rev-parse HEAD)
/usr/bin/python3.12 -I -s -B deployment/build_freshness_runtime.py \
  --wheelhouse /bezwzgledna/sciezka/wheelhouse \
  --destination /bezwzgledna/sciezka/kandydat-runtime \
  --environment-id noble-cp312-trial-001 \
  --release-revision "$trial_revision" \
  --manifest /bezwzgledna/sciezka/runtime-manifest.json
```

Docelowy katalog i manifest muszą być nowe. Zachować stdout/stderr,
`build-probe.json`, unsigned manifest, lock i inwentarz natywny. Sonda musi
potwierdzić izolację, dokładne dystrybucje, ABI, prawdziwe Ed25519 oraz
`psycopg.pq.__impl__ == 'binary'`. Nie otwiera ona połączenia z PostgreSQL.

## 3. Niezależny podpis TEST ONLY

Inna rola operatora porównuje źródła, hashe, wersje, konfigurację venv,
inwentarz oraz pochodzenie systemowego interpretera i bibliotek. Własność
kandydata nie jest podpisem ani zgodą na produkcję. Wyłącznie do tej VM
można utworzyć nowy klucz fixture na odrębnym stanowisku:

```sh
umask 077
openssl genpkey -algorithm ED25519 -out trial-authority.pem
openssl pkey -in trial-authority.pem -pubout -outform DER -out trial-authority.der
python3 -I -s -B - <<'PY'
from pathlib import Path
raw = Path('trial-authority.der').read_bytes()
assert len(raw) == 44 and raw[:12].hex() == '302a300506032b6570032100'
Path('trial-authority.pub').write_bytes(raw[12:])
PY
openssl pkeyutl -sign -rawin -inkey trial-authority.pem \
  -in runtime-manifest.json -out runtime-manifest.sig
```

Nie kopiować klucza prywatnego do bundle ani VM wykonawczej. Zaufany klucz
publiczny dostarczyć osobnym kanałem i porównać jego odcisk. Zachować decyzję
przeglądu. Nie używać kluczy produkcyjnych. Każdy zmieniony bajt manifestu
wymaga nowego podpisu.

## 4. Weryfikacja, publikacja i binding

W VM operator umieszcza zatwierdzone wejścia w nowym chronionym katalogu,
np. `/var/lib/cryptohunter-runtime-input/trial-001`. Pliki mają root jako
właściciela, katalogi 0755 i pliki 0644; symlinki zachować dokładnie według
manifestu. Nie naprawiać rekurencyjnie zastanego, obcego venv.
Katalog bazowy `/opt/cryptohunter/freshness-runtime` przygotować osobno.
Publiczny klucz raw32 instalować pod
`/etc/cryptohunter/freshness-runtime-authority.pub`, root:root, 0644,
z chronionym łańcuchem przodków. Administrator nie uznaje klucza dołączonego
przez nieznany bundle za automatycznie zaufany.

Najpierw odczytowa kontrola, potem osobna publikacja operatora:

```sh
sudo /usr/bin/python3 -I -s -B /srv/cryptohunter-trusted/deployment/freshness_runtime.py \
  verify --runtime /var/lib/cryptohunter-runtime-input/trial-001/runtime \
  --manifest /var/lib/cryptohunter-runtime-input/trial-001/runtime-manifest.json \
  --signature /var/lib/cryptohunter-runtime-input/trial-001/runtime-manifest.sig \
  --trusted-key /etc/cryptohunter/freshness-runtime-authority.pub \
  --environment-id noble-cp312-trial-001 --revision "$trial_revision"
sudo /usr/bin/python3 -I -s -B /srv/cryptohunter-trusted/deployment/freshness_runtime.py \
  publish --runtime /var/lib/cryptohunter-runtime-input/trial-001/runtime \
  --manifest /var/lib/cryptohunter-runtime-input/trial-001/runtime-manifest.json \
  --signature /var/lib/cryptohunter-runtime-input/trial-001/runtime-manifest.sig \
  --trusted-key /etc/cryptohunter/freshness-runtime-authority.pub \
  --environment-id noble-cp312-trial-001 --revision "$trial_revision"
```

Podpisany manifest runtime podaje installation_path. Konfiguracja
`/etc/cryptohunter/freshness-runtime.json` ma dokładnie pola:

```json
{
  "schema": 1,
  "environment_id": "noble-cp312-trial-001",
  "manifest_sha256": "<SHA-256 dokładnych podpisanych bajtów manifestu>",
  "source_revision": "<zatwierdzona rewizja 40 hex>",
  "executable": "/opt/cryptohunter/freshness-runtime/releases/noble-cp312-trial-001/bin/python3.12",
  "base_executable": "/usr/bin/python3.12",
  "base_executable_sha256": "<SHA-256 z external_files bazowego interpretera>"
}
```

Zastąpić wszystkie znaczniki rzeczywistymi zatwierdzonymi wartościami. Binding
jest chronioną konfiguracją administratora; podpisany manifest aplikacji v2
musi zawierać dokładnie ten sam obiekt `runtime` bez pola `schema`. Jednostkę
zmaterializować zastępując pojedynczy token `@FRESHNESS_RUNTIME_PYTHON@`
dokładną ścieżką executable, przed obliczeniem hasha i podpisaniem bundle.
Inwentarz aplikacji wyeksportować z aktualnego `REQUIRED_FILES` przez AST
według istniejącej procedury; nie zakładać historycznej liczby plików.
Wymagane scope to `cryptohunter.freshness.substrate.v2`, schema_version=2,
a podpis aplikacji pochodzi z jej odrębnie zatwierdzonego autorytetu.

## 5. Próba dodatnia i odmowy

Uruchomić odczytowy preflight instalatora według jego `--help` i istniejącego
runbooka. Porównać inwentarze hosta przed/po: bez nowych plików, kont, klastra
lub restartów. Dopiero po spełnieniu wszystkich warunków operator wykonuje
instalację aplikacji w VM. Zachować receipt, status systemd i skuteczne ustawienia
sandboxu. Powtórzyć instalację identycznych materiałów i porównać dane guard.

Przed startem usługi sprawdzić `systemd-analyze verify` dla zmaterializowanej
jednostki. Potem operator zachowuje dokładne D-Bus ExecStart/ExecStartPre,
MainPID/ExecMainPID, argv, `/proc/PID/exe`, inode/hash, UID/GID, cgroup i wynik
pidfd. Pełna kwalifikacja może nadal odmówić z powodu odrębnych blokerów produktu;
nie zastępować jej atrapą.

Każdy poniższy przypadek wykonywać na osobnym snapshotcie/kopii fixture,
nigdy na aktywnym wydaniu. Oczekiwana jest odmowa przed wykonaniem zmienionego
runtime i brak mutacji preflight:

- brak wheel, dodatkowy wheel/pakiet, błędny hash oraz nieprawidłowy podpis;
- obcy `.pth`, user-site, zmieniony pyvenv.cfg, binarka, symlink lub tryb pliku;
- inny ABI, environment-id, manifest albo rewizja;
- inny interpreter jednostki, brak/zmiana bootstrapu, drop-in lub inny argv;
- podmieniony lub kończący się proces, PID reuse, inny inode/hash/cgroup;
- przerwanie przed rename i po rename przed usunięciem journala;
- powtórzenie identycznego wydania oraz próba nadpisania innymi materiałami;
- drugie zatwierdzone environment-id z zachowaniem pierwszego wydania;
- aktualizacja systemowego interpretera/stdlib/native: istniejący podpis ma
  przestać wystarczać; wymagany ponowny przegląd, bez automatycznej aktualizacji hashy.

Przerwaną publikację odzyskiwać wyłącznie po zachowaniu dowodów i niezależnej
weryfikacji; nie usuwać automatycznie poprzednich wydań ani danych. Zgon procesu
w trakcie odczytu nie może dać PASS. W raporcie rozdzielić build/importy,
instalację, systemd, PostgreSQL i pełny produkt. Fizyczne lifecycle/reboot/update
pozostają osobną ścieżką dowodów i nie są zaliczane tą procedurą.

## Osobny test kontenerowy buildera i podpisu

Wersjonowany `tests/deployment/freshness_runtime_noble_trial.py` odtwarza
racjonalnie ograniczoną próbę native build → podpis TEST ONLY → verify → publish,
rzeczywiste importy i odmowy. Działa wyłącznie jako root **wewnątrz nowego
jednorazowego kontenera**, z `/.dockerenv`, Noble i jawnym
`CRYPTOHUNTER_DISPOSABLE_NOBLE_TRIAL=1`; odmawia zastanych wydań.
Nie uruchamiać go na fizycznym hoście ani w VM z danymi.

Przed odłączeniem sieci przygotować obraz z oficjalnymi pakietami
`python3.12-venv`, `binutils` i `ca-certificates`; zachować faktyczny digest
obrazu i wersje APT. W nowym kontenerze `--network none` umieścić przejrzane
`deployment/` i `tests/deployment/` w chronionej kopii `/reviewed`, jako root,
oraz kompletny wheelhouse pod `/wheels`. Po skopiowaniu materiałów uruchomić:

```sh
CRYPTOHUNTER_DISPOSABLE_NOBLE_TRIAL=1 /usr/bin/python3.12 -I -s -B \
  /reviewed/tests/deployment/freshness_runtime_noble_trial.py \
  --wheelhouse /wheels --revision "$trial_revision" \
  --report /tmp/noble-checks.json --base-image "$trial_image_digest"
```

Parametr `--base-image` zapisuje obserwację operatora; nie atestuje obrazu.
Zachować raport z kontenera, następnie go usunąć. Prywatny klucz fixture
pozostaje wyłącznie w tym jednorazowym kontenerze. Wynik nie zastępuje systemd,
PostgreSQL ani próby VM.
