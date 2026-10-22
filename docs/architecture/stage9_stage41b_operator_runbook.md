# Stage 41B — procedura operatora na fizycznym Dellu

Ta procedura dotyczy wyłącznie syntetycznej kwalifikacji PR
[#3202](https://github.com/Dudzian/Dudzian/pull/3202) na Dell Latitude 5490
z Ubuntu 24.04 x86_64 i CPython 3.12. Przygotowanie środowiska, weryfikacja
instalacji, preflight i wykonanie testu są odrębnymi operacjami. Żaden krok
przygotowania nie uruchamia signerów ani `--execute`. Nie wykonuje się rebootu.

## 1. Przypnij zatwierdzony commit i przygotuj bundle bez roota

Z ostatniego przejrzanego commita PR odczytaj pełny Git SHA oraz niezależnie
opublikowany SHA256 pliku `scripts/prepare_stage41b_environment.py`.
Poniższe wartości są znacznikami do zastąpienia, nie gotowymi hashami:

```bash
STAGE41B_COMMIT='REPLACE_WITH_REVIEWED_PR3202_FULL_COMMIT_SHA'
STAGE41B_BOOTSTRAP_SHA256='REPLACE_WITH_INDEPENDENTLY_REVIEWED_INSTALLER_SHA256'
STAGE41B_SOURCE="/home/dudzian/stage41b-source-$STAGE41B_COMMIT"
STAGE41B_BUNDLE="/home/dudzian/stage41b-bundle-$STAGE41B_COMMIT"
STAGE41B_INSTALL="/opt/cryptohunter-stage41b/$STAGE41B_COMMIT"
STAGE41B_REPORTS="/home/dudzian/stage41b-reports-$STAGE41B_COMMIT"
```

Nie uznawaj hasha obliczonego wyłącznie z aktualnego pliku w katalogu domowym
za niezależne potwierdzenie jego zaufania. Porównaj wartości z przejrzanym
commitem i raportem PR; aktualizacja PR wymaga ponownego przypięcia SHA,
wyników CI i digestu instalatora. Nie używaj ruchomego `main`, `HEAD`, numeru
PR lub nazwy brancha jako identyfikatora instalacji.

Przygotuj osobny checkout dokładnego commita; te polecenia nie dotyczą
`/home/dudzian/CryptoHunter`:

```bash
git clone --no-checkout https://github.com/Dudzian/Dudzian.git "$STAGE41B_SOURCE"
git -C "$STAGE41B_SOURCE" fetch origin "$STAGE41B_COMMIT"
git -C "$STAGE41B_SOURCE" checkout --detach "$STAGE41B_COMMIT"
git -C "$STAGE41B_SOURCE" rev-parse HEAD
/usr/bin/python3 -B "$STAGE41B_SOURCE/scripts/prepare_stage41b_environment.py" \
  --build --source "$STAGE41B_SOURCE" \
  --commit "$STAGE41B_COMMIT" --bundle "$STAGE41B_BUNDLE"
```

Builder wymaga czystego, zweryfikowanego źródła i obecności wskazanego commita.
Nie wykonuje `checkout`, `reset`, `clean`, zmiany właściciela ani zapisu
w `/home/dudzian/CryptoHunter`. Checkout i bundle muszą być nowymi katalogami
poza oryginalnym repozytorium, a wynik `rev-parse HEAD` musi odpowiadać
`STAGE41B_COMMIT`. Jeśli pobranie commita lub weryfikacja źródeł nie powiedzie
się, zatrzymaj ten krok. Nie naprawiaj tego przez resetowanie roboczego
katalogu użytkownika. Istniejące czyste źródło można wskazać przez `--source`
wyłącznie, jeżeli ma już dokładny zaakceptowany HEAD; nie wykonuj w nim
`fetch`, `checkout`, `reset` lub zmiany uprawnień na potrzeby tej procedury.

Bundle wiąże wybrane źródła przez Git blob/tree/commit proof z dokładnym SHA
oraz zawiera wyłącznie wheels przypięte w
`deployment/stage41b-wheel-lock.json` i `deployment/stage41b-requirements.lock`.
Pobieranie odbywa się bez uprawnień roota z określonych adresów oficjalnych
artefaktów PyPI, z kontrolą SHA256. Nie ma source distributions, resolvera
dobierającego nowe wersje ani wykonywania build hooków. Zależności systemowe
Ubuntu, takie jak systemd, D-Bus i GNOME Keyring, muszą być wcześniej dostępne;
ich brak jest BLOCKED, bez automatycznego `apt install`.

## 2. Zainstaluj wyłącznie przypięty bootstrap i zweryfikowane artefakty

Uprzywilejowany instalator musi najpierw znaleźć się w nowym, root-owned
katalogu staging, a SHA256 **tej skopiowanej wersji** musi odpowiadać
`STAGE41B_BOOTSTRAP_SHA256` odczytanemu z przejrzanego PR. Nie wykonuj
`sudo python /home/dudzian/CryptoHunter/...` ani `sudo pip install`.

Instalator działa jako osobny program stdlib z systemowym Pythonem w trybie
izolowanym. Zanim użyje artefaktów, weryfikuje dokładny commit, Git proof,
zamknięty zestaw ścieżek, lock oraz hashe wheels. Instalacja root działa
offline; nie pobiera zależności ani nie wykonuje pip, sdist lub kodu z wheel
podczas jego rozpakowywania. Zaufanym wejściem wykonawczym jest wyłącznie
bootstrap, którego root-staged bytes przeszły niezależną kontrolę SHA256.

Docelowy layout jest stały:

```text
/opt/cryptohunter-stage41b/<full-commit-sha>/
    source/
    venv/
    installation.json
    bundle.json
    artifacts/wheels/
```

Istniejąca instalacja o tej nazwie nie jest nadpisywana. Instalacja nie używa
produkcyjnych konfiguracji, keyringów ani sekretów; nie instaluje units
`deployment/systemd`, nie zmienia sysusers i nie uruchamia signerów. Kod,
interpreter, zależności i manifesty mają chronione pochodzenie i uprawnienia;
oryginalny katalog domowy pozostaje bez zmian. Polecenia bootstrapu poniżej
należy wykonywać dopiero z wartościami przypiętymi do aktualnego commita PR.

Każde użycie sudo poniżej dotyczy wskazanej operacji i konkretnego katalogu;
procedura nie wymaga zmiany sudoers ani nieograniczonego sudo. Najpierw sprawdź
format przypiętych wartości, utwórz nowy root-private staging i skopiuj jeden
publiczny plik instalatora:

```bash
if ! [[ "$STAGE41B_COMMIT" =~ ^[0-9a-f]{40}$ && "$STAGE41B_BOOTSTRAP_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'BLOCKED: replace placeholders with independently reviewed hashes'
  exit 1
fi
if ! STAGE41B_ROOT_STAGE="$(sudo /usr/bin/mktemp -d /run/cryptohunter-stage41b-bootstrap.XXXXXXXX)"; then
  printf '%s\n' 'BLOCKED: root staging creation failed'
  exit 1
fi
if ! [[ "$STAGE41B_ROOT_STAGE" =~ ^/run/cryptohunter-stage41b-bootstrap\.[A-Za-z0-9]{8}$ ]]; then
  printf '%s\n' 'BLOCKED: unexpected staging path'
  exit 1
fi
if ! sudo /usr/bin/install -m 0500 -- \
  "$STAGE41B_SOURCE/scripts/prepare_stage41b_environment.py" \
  "$STAGE41B_ROOT_STAGE/bootstrap.py"; then
  printf '%s\n' "BLOCKED: inspect root staging $STAGE41B_ROOT_STAGE"
  exit 1
fi
if ! STAGE41B_STAGED_SHA256="$(sudo /usr/bin/sha256sum -- "$STAGE41B_ROOT_STAGE/bootstrap.py")"; then
  printf '%s\n' "BLOCKED: inspect root staging $STAGE41B_ROOT_STAGE"
  exit 1
fi
if [ "${STAGE41B_STAGED_SHA256%% *}" != "$STAGE41B_BOOTSTRAP_SHA256" ]; then
  printf '%s\n' "BLOCKED: staged installer SHA256 mismatch; do not execute; inspect $STAGE41B_ROOT_STAGE"
  exit 1
fi
```

Dopiero po zgodności hasha uruchom tę chronioną kopię. Installer odczytuje
bundle jako niezaufane dane, uwierzytelnia Git proof i wheels, a następnie
przenosi zweryfikowane snapshoty bytes do root-owned instalacji. Rozpakowuje
te same zweryfikowane bajty; nie ponownie odczytany, zmienialny plik z home.

```bash
sudo /usr/bin/python3 -I -S -B "$STAGE41B_ROOT_STAGE/bootstrap.py" \
  --install --bundle "$STAGE41B_BUNDLE" --commit "$STAGE41B_COMMIT"
```

Nie kontynuuj po błędzie instalacji. Zachowaj jej raport i informacje
o utworzonym katalogu; nie nadpisuj częściowej instalacji ani nie usuwaj jej
rekurencyjnie, aby wymusić powtórzenie. Po pomyślnym zakończeniu usuń jedynie
utworzony w tym kroku bootstrap i jego pusty katalog:

```bash
sudo /usr/bin/rm -- "$STAGE41B_ROOT_STAGE/bootstrap.py"
sudo /usr/bin/rmdir -- "$STAGE41B_ROOT_STAGE"
```

Po zakończeniu instalacji jawnie sprawdź jej kompletność i integralność:

```bash
/usr/bin/mkdir -m 0700 -- "$STAGE41B_REPORTS"
sudo /usr/bin/python3 -I -S -B \
  "$STAGE41B_INSTALL/source/scripts/prepare_stage41b_environment.py" \
  --verify-installed --commit "$STAGE41B_COMMIT" > "$STAGE41B_REPORTS/installation.json"
/usr/bin/cat -- "$STAGE41B_REPORTS/installation.json"
```

Pozytywny wynik oznacza weryfikację środowiska, a nie fizyczną kwalifikację
kluczy. Nie zastępuje następnego preflight ani 19 kontroli Stage 41B.
Kolejny przebieg tego samego przejrzanego commita może wykorzystać istniejące
środowisko wyłącznie po ponownym `--verify-installed`; nie powtarzaj instalacji
nad istniejącym katalogiem. Aktualizacja systemowego Pythona lub zmiana
jakiegokolwiek przypiętego pliku może unieważnić integralność. Taki BLOCKED
wymaga przeglądu i nowego przygotowania, a nie edycji manifestu lub wyłączenia
kontroli. Samo istnienie katalogu `/opt` nie oznacza zaufanego środowiska.

## 3. Uruchom odczytowy preflight

Uruchamiaj harness z chronionego `source`, jego przypiętym interpreterem,
bez dziedziczenia Python path ani user site packages:

```bash
cd "$STAGE41B_INSTALL/source"
sudo "$STAGE41B_INSTALL/venv/bin/python" -I -B \
  -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module("scripts.qualify_stage41b_signer_isolation", run_name="__main__")' \
  "$STAGE41B_INSTALL/source" --check > "$STAGE41B_REPORTS/preflight.json"
/usr/bin/cat -- "$STAGE41B_REPORTS/preflight.json"
```

`--check` nie tworzy jednostek, kluczy, keyringów ani katalogów testu.
Sprawdza system, chroniony kod, interpreter i zależności oraz brak kolizji
z jednostkami i ścieżkami credential. Nazwy produkcyjnych signerów muszą być
niezainstalowane, niezamaskowane, bez drop-inów, niezaładowane i nieaktywne.
Istniejącej produkcji nie zatrzymuje się, aby zrobić miejsce testowi.

Wrapper dodaje do `sys.path` jedynie wskazane, już zweryfikowane chronione
`source`. Samo `-I -m` nie znajduje modułu z bieżącego katalogu; nie zastępuj
wrappera dziedziczeniem `PYTHONPATH`, `sudo -E` lub user site packages.
Nie dodawaj `-S` do interpretera venv: w CPython 3.12 zmienia to rozpoznanie
środowiska i dostępność jego przypiętych zależności.

Wynik preflight nie jest `STAGE41B_TEST_ONLY_PASS`; niewykonane fizyczne
granice nadal pozostają BLOCKED. Również poprawny `--check` zwraca kod `2`
i zbiorczy status BLOCKED. Przesłanki są gotowe wyłącznie, gdy
`boundaries.host_preflight.status` wynosi `PASS`; pozostałe fizyczne kontrole
mają w tym trybie `PHYSICAL_CONTROL_NOT_EXECUTED`. Przy zablokowanym
`host_preflight` zatrzymaj procedurę. Nie osłabiaj `_trusted_code()` ani
uprawnień źródeł w katalogu domowym.

## 4. Świadomie uruchom fizyczny test

Operator wykonuje następne polecenie dopiero po przejrzeniu pozytywnego
preflight i świadomym wyborze uruchomienia syntetycznych units na Dellu.
Przygotowanie i preflight nie wywołują tego kroku automatycznie.

```bash
sudo "$STAGE41B_INSTALL/venv/bin/python" -I -B \
  -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module("scripts.qualify_stage41b_signer_isolation", run_name="__main__")' \
  "$STAGE41B_INSTALL/source" --execute > "$STAGE41B_REPORTS/execution.json"
/usr/bin/cat -- "$STAGE41B_REPORTS/execution.json"
```

Harness tworzy dwa tymczasowe units z odrębnymi UID/GID, mount/IPC/network
namespaces i `NoNewPrivileges`. Korzysta z oryginalnego custody i rzeczywistego
GNOME Secret Service. Podpisuje wyłącznie z góry ustalone syntetyczne komunikaty
z bieżącym run-id i rolą. Nie wykonuje produkcyjnego SIGN, BIND, provisioning,
enrollmentu ani rebootu. Nie uruchamiaj root-only pytest na głównym Ubuntu.

Przekierowanie stdout wykonuje zwykła powłoka operatora; root nie zapisuje
raportów w jego home. Zachowaj publiczny JSON i jego `run_id`. Raport musi zawierać osobny wynik
każdej z 19 wymaganych granic oraz wynik sprzątania. Tylko kompletne fizyczne
PASS wszystkich wymaganych kontroli wraz ze sprzątaniem może dać
`STAGE41B_TEST_ONLY_PASS`. FAIL, BLOCKED, przerwany przebieg lub pominięcie
wyklucza ten wynik. Test obu magistral wymaga obu żywych usług przed próbami
i po nich; samo ENOENT nie jest dowodem bezpieczeństwa.

## 5. Sprawdź pozostałości i zachowaj raport

Odczytaj dokładny `run_id` z JSON zakończonego lub przerwanego przebiegu:

```bash
STAGE41B_RUN_ID="$(/usr/bin/python3 -I -S -c 'import json, sys; print(json.load(open(sys.argv[1]))["run_id"])' "$STAGE41B_REPORTS/execution.json")"
sudo "$STAGE41B_INSTALL/venv/bin/python" -I -B \
  -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module("scripts.qualify_stage41b_signer_isolation", run_name="__main__")' \
  "$STAGE41B_INSTALL/source" --inspect-cleanup --run-id "$STAGE41B_RUN_ID" \
  > "$STAGE41B_REPORTS/cleanup.json"
/usr/bin/cat -- "$STAGE41B_REPORTS/cleanup.json"
```

Inspekcja jest odczytowa: nie zatrzymuje jednostek i nie usuwa katalogów.
Zachowaj jej JSON razem z raportem testu, SHA commita, wynikiem weryfikacji
instalacji i preflight. Do podsumowania dołącz 19 wyników, błędy lub
pominięcia oraz szczegółowy wykaz pozostałości. Nie publikuj zrzutów keyringu,
środowiska procesu ani prywatnego materiału.

Odczytowy wynik `PASS` inspekcji oznacza jedynie brak oczekiwanych pozostałości;
nie podnosi wcześniejszego BLOCKED lub FAIL do fizycznego PASS. Jeśli nagłe
`SIGKILL` lub utrata zasilania uniemożliwiły powstanie raportu z run-id, nie
zgaduj go ani nie usuwaj zasobów przez wildcard. Wymagany jest osobny przegląd
metadanych i pochodzenia przez administratora.

Automatyczny cleanup identyfikuje własną jednostkę przez przypięte pochodzenie,
Description z run-id, InvocationID, cgroup i proces; nie zatrzymuje niczego
wyłącznie na podstawie nazwy. Wygasły PID lub zastąpiona jednostka nie są
uprawnieniem do zabicia nowego procesu. Niejednoznaczne zasoby są zachowywane
z BLOCKED i instrukcją inspekcji. Nie wykonuj zbiorczego `systemctl stop`,
`reset-failed`, `rm -rf`, resetu keyringu lub rebootu w celu uzyskania PASS.
Nie powtarzaj testu, zanim pochodzenie pozostałości nie zostanie rozstrzygnięte.

Pozytywny wynik syntetyczny nadal nie dowodzi niezależnej kontroli
produkcyjnych kluczy. Stage 9 pozostaje `IN_PROGRESS / NO-GO`; istniejące
blokady private-control attestation i cross-role native keyring nie są
usuwane. Zakres dowodu i pozostałe wymagania opisuje
[dokument architektoniczny Stage 41B](stage9_stage41b_physical_signer_isolation.md).
