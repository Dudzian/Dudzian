# Stage 9 V5d11 — izolacja i diagnostyka profilu A

V5d11 przygotowuje i mierzy prywatny widok wyprowadzony z istniejącego receipt
V5d10. Osobny receipt uruchomieniowy wiąże bajty źródeł i aprobatę
materializacji z dokładnym właścicielem, grupą i trybami kopii. Profil V5d1
zachowuje dotychczasowe wymagania i kod wyjścia `2`. Bootstrap V5d11 udostępnia
pomiar oraz oddzielną diagnostykę z zamkniętym zestawem sond. Polecenie `GO`
zawsze odrzuca. [Kontrakt V5d10](stage9_v5d10_observer_preparation.md) ·
[Zgodność z V5d1](stage9_v5d11_v5d1_compatibility.md)

## 1. Przygotowanie i niezależny przegląd

Umieść przejrzane źródła z dokładnej rewizji PR w chronionym katalogu
`/opt/stage9-reviewed`. Wszystkie przodki i zawartość używana przez
uprzywilejowany operator mają należeć do roota i być chronione przed zapisem
grupy oraz innych użytkowników. Przykład zakłada prywatny katalog wejść
`/root/stage9-private` z trybem `0700`. Ustaw tryb `0600` na plikach wejść.
Potrzebne są chronione narzędzia Ubuntu 24.04, w tym CPython 3.12, OpenSSL,
kompilator ze statyczną libc, parser AppArmor oraz `/usr/bin/readelf` z binutils.

Przygotuj świeży eksport z zaufanego checkoutu. Podaj pełny SHA zatwierdzonego
commita z niezależnego przeglądu. Wymagaj pustej listy z `find` i sprawdź
własność i tryby przodków wskazane przez `namei`. `-B` wyłącza zapis cache
Pythona. Zaufanie do źródeł opiera się na eksporcie zatwierdzonego commita
i ochronie katalogu przed zmianami. Uprzywilejowany operator ładuje zamkniętą
listę plików `.py` z tego drzewa bez wyszukiwania modułów i cache w checkout.

```bash
(
  set -euo pipefail
  read -r -p 'Bezwzględna ścieżka zaufanego checkoutu: ' checkout
  read -r -p 'Pełny SHA niezależnie zatwierdzonego commita: ' revision
  [[ $checkout = /* && $revision =~ ^[0-9a-f]{40}$ ]]
  [[ $(git -C "$checkout" cat-file -t "$revision") = commit ]]
  git -C "$checkout" fsck --full --strict
  target=/opt/stage9-reviewed
  sudo /usr/bin/mkdir -m 0755 -- "$target"
  git -C "$checkout" archive "$revision" -- scripts deployment/stage9_v5d11 |
    sudo /usr/bin/tar -x --no-same-owner --no-same-permissions -C "$target"
  sudo /usr/bin/namei -l -- "$target"
  findings=$(sudo /usr/bin/find "$target" -xdev \( ! -uid 0 -o -perm /022 \
    -o -type l -o -name '__pycache__' -o -name '*.pyc' -o -name '*.pyo' \) -print)
  test -z "$findings" || { printf '%s\n' "$findings"; exit 2; }
  (cd "$target" && sha256sum --strict --check deployment/stage9_v5d11/SHA256SUMS)
)
```

Utwórz `inputs.json` według przykładu. Wprowadź niezależne piny receipt, aprobaty
V2, klucza publicznego recenzenta oraz polityk pomiarowych. Wybierz osobny UID
i GID dedykowanego konta diagnostycznego bez działających procesów,
dodatkowych członków grupy i możliwości logowania. `root` wskazuje dokładną
ścieżkę zapisaną w istniejącym receipt. `data_files` wskazuje tylko przypięte
źródła. Dla obrazów wymagany jest dodatkowy bezwzględny `scratch` na lokalnym
ext4 o trybie `0700`, należący do roota. Operator przygotowuje osobne fixture
raw i QCOW2 bez backing file, z własnymi drogami logicznymi i pinami. Oryginalne
obrazy VM pozostają źródłami tylko do odczytu. Pusty zakres danych jest
wystarczający dla pomiaru samego układu, bez późniejszej diagnostyki QEMU.
[Przykładowe wejścia](../../deployment/stage9_v5d11/operator-inputs.example.json)

Zbuduj bootstrap w chronionym katalogu. Polecenie kompiluje kod C i odczytuje
wyłącznie skróty binarki i bajtów BPF.

```sh
/usr/bin/python3.12 -I -S -B /opt/stage9-reviewed/scripts/stage9_v5d11_operator.py \
  build /root/stage9-private/bootstrap
```

Przekaż recenzentowi wynik, dokładne źródła i pakiet pomiarowych polityk.
Render i kompilacja offline nie ładują polityk na hosta. Instalacja wymaga
oddzielnej zgody operatora. Tryb `load-policy` stosuje `add` i sprawdza eksport
jądra. Przed dodaniem wariantu diagnostycznego potrzebny jest niezależny
przegląd jego odrębnych bajtów. [Polecenia renderowania i polityki](../../deployment/stage9_v5d11/README.md)

```sh
sudo /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  /usr/bin/python3.12 -I -S -B /opt/stage9-reviewed/scripts/stage9_v5d11_operator.py \
  load-policy /root/stage9-private/inputs.json
```

## 2. Fizyczny pomiar widoku

W inwentaryzacji odczytaj wersję jądra i ustawienie AppArmor. Zachowaj oba
odczyty w prywatnym pakiecie.

```sh
uname -srmo
sudo /usr/bin/cat /proc/sys/kernel/apparmor_restrict_unprivileged_unconfined
```

Oczekiwana wartość przed `measure` i `diagnose` to `0`. Wartość `1` może
spowodować dołączenie etykiety do `unconfined` po zrzuceniu capabilities.
Bootstrap wymaga pojedynczej dokładnej etykiety `enforce`. Wartość `1`,
brak odczytu lub nieprawidłowa wartość blokują próbę. Wartość `0` pozwala
ocenić tę bramkę wejściową. Faktyczne przejście i etykietę sprawdza bootstrap.
Każda zmiana konfiguracji zabezpieczeń hosta ma własną procedurę.

Dla jąder z zachowaniem źródeł Ubuntu Noble GA 6.8 lub HWE `7.0.0-43.43~24.04.1`
odnotuj osobną przeszkodę. AppArmor może odmówić `stat` obiektu z wewnętrznego
montowania `nsfs` po przejściu do `enforce` i zakończyć bootstrap przez
`BOOTSTRAP_NAMESPACE_STAT` przed zapisaniem pomiaru. Ustal zgodność dokładnego
jądra hosta przed dopuszczeniem prób. [Obsługa ścieżki w AppArmor w HWE 7.0.0-43](https://git.launchpad.net/~ubuntu-kernel/ubuntu/+source/linux/+git/noble/tree/security/apparmor/path.c?id=60e65196d87a3b82950ceb158a08953b6468c785#n97)

Po dopuszczeniu samego pomiaru uruchom polecenie z nowym katalogiem wynikowym.

```sh
sudo /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  /usr/bin/python3.12 -I -S -B /opt/stage9-reviewed/scripts/stage9_v5d11_operator.py \
  measure /root/stage9-private/inputs.json /root/stage9-private/measurement
```

Bootstrap potwierdza osobne przestrzenie mount, PID i network, root RO,
prywatny procfs, zamknięcie deskryptorów, brak capabilities, `NoNewPrivs`,
AppArmor, BPF oraz dokładny socket RO. Dla każdego obrazu sprawdza immutable,
ext4, chroniony obiekt i `dev:ino:size` po wejściu do widoku. Do katalogu
wynikowego trafiają `mountinfo`, `measurement.json`, kandydatura launchera
V5d10 i `runtime-receipt.candidate.json`. Pomiar kończy się bez oczekiwania
na podpis. Zachowaj cały zestaw w prywatnym obiegu.

## 3. Odrębna autoryzacja diagnostyki

Wygeneruj statyczny raport ABI z już przypiętych wejść. Przygotuj
`diagnostic.json` z dokładnym hashem wcześniejszego pomiaru, osobnego receipt,
statycznego raportu, payloadu, niezależnie przejrzanej implementacji RO
demona i renderowanej oraz skompilowanej polityki diagnostycznej. Skonfiguruj
domenę testową oraz oddzielny preprovisioned fixture demona o zgodnych
bajtach, konfiguracji i odrębnych przestrzeniach nazw. Przygotowanie i
utrzymanie tego fixture należy do operatora stanowiska i jest poza działaniem
kolektora. [Przykładowa specyfikacja](../../deployment/stage9_v5d11/diagnostic-inputs.example.json) ·
[Wymagania fixture i raportu ABI](stage9_v5d11_libvirt_loader_qualification.md)

Po osobnej zgodzie na instalację polityki diagnostycznej użyj jednego
polecenia. Profil diagnostyczny ma osobne etykiety. Wariant przejścia
`px` zachowuje jedynie jawnie ustawione środowisko diagnostyczne z
`LD_DEBUG`. Uprawnienie `readby` pozwala zakwalifikowanemu hostowemu
kolektorowi odczytać własne procesy przez procfs. Zakazy śledzenia i
przypisanie dedykowanego UID pozostają egzekwowane. Wariant nie nadaje odczytu
źródeł Observera ani dodatkowych bibliotek.

```sh
sudo /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  /usr/bin/python3.12 -I -S -B /opt/stage9-reviewed/scripts/stage9_v5d11_operator.py \
  load-policy /root/stage9-private/inputs.json \
  --diagnostic-inputs /root/stage9-private/diagnostic.json
```

Niezależny recenzent przygotowuje obiekt `review` z dokładnie sześcioma polami.
Wartości `profile` i `decision` to
`stage9-v5d11-diagnostic-authorization-v2` i
`AUTHORIZE_NATIVE_DIAGNOSTIC_ONLY`. `spec_sha256` wiąże dokładne bajty
`diagnostic.json`. `nonce` to losowy, jednorazowy identyfikator z 64 małych
znaków hex. `expires_utc` to całkowity czas wygaśnięcia Unix, maksymalnie
godzinę po weryfikacji. `host_boot_id` ma wartość kanonicznego UUIDv4 z
`/proc/sys/kernel/random/boot_id`, przekazaną recenzentowi w prywatnym obiegu
bez końcowego LF. Recenzent podpisuje kanoniczne bajty `review` Ed25519
według serializatora V5d10, bez końcowego znaku nowej linii. Odpowiedź zawiera
tylko `review` i `signature_hex`. Klucz publiczny w DER SubjectPublicKeyInfo
musi odpowiadać niezależnemu pinowi w manifeście. Narzędzie pomiarowe nie tworzy
aprobaty. Zgoda na wykonanie jest oddzielna od dodania polityki.

Po otrzymaniu podpisu operator uruchamia nową sesję, ponownie sprawdza widok
i zużywa nonce przed pierwszym działaniem diagnostycznym. Użycie innego
katalogu specyfikacji nie zmienia powiązanej ścieżki manifestu ani miejsca
zapisu znacznika. Nieudana próba po zużyciu nonce wymaga nowej zgody.
Marker pozostaje w trwałym, chronionym katalogu manifestu. W okresie ważności
zgody zachowaj katalog i znaczniki, również podczas procedur odtwarzania.
Zgody V1, zgody z innego rozruchu i próby po ponownym uruchomieniu hosta
wymagają nowego podpisu V2. Bajty specyfikacji diagnostycznej zachowują
dotychczasowy schemat.

```sh
sudo /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  /usr/bin/python3.12 -I -S -B /opt/stage9-reviewed/scripts/stage9_v5d11_operator.py \
  diagnose /root/stage9-private/inputs.json /root/stage9-private/diagnostic.json \
  /root/stage9-private/diagnostic-authorization.json /root/stage9-private/diagnostic-run
```

## 4. Kwalifikacja rzeczywistych dowodów

Kolektor zapisuje `native-evidence.json`, jego artefakty źródłowe i ocenę
kwalifikatora. Wiąże tożsamość oraz konfigurację demona, odczyt protokołu i
niedostępność zapisu w odrębnym fixture, PID i etykietę potomka, mapowania
jądra, surowe zdarzenia loadera, tablice symboli i skróty ELF. Status
`EVIDENCE_CONSISTENT_REVIEW_REQUIRED` oznacza spójność z przypięciami i
potrzebę osobnego przeglądu dowodów. Fizyczne egzekwowanie polityk, rzeczywisty
loader, libvirt i qemu-img pozostają `NOT_TESTED` w tej zmianie.
[Powtórna weryfikacja offline](stage9_v5d11_libvirt_loader_qualification.md)

`STAGE_9=IN_PROGRESS` · `OBSERVER_3A=BLOCKED` · `STAGE_3B=NOT_AUTHORIZED` ·
`PRODUCTION=NO-GO`.
