# Stage 9 V5d10 — kwalifikacja preflight Observera

V5d10 przygotowuje osobny profil A dla odczytowego preflight V5d1. Builder A
sprawdza cztery archiwa, przypięte wejścia, statyczne ABI i pełny plan katalogu.
Recenzent ocenia niezależnie hash dokumentu aprobaty. Launcher weryfikuje
wystawiony receipt w trybie `verify` bez importu pakietów Observera. Tryb
`preflight` wymaga osobnej autoryzacji i uprzednio przygotowanej izolacji
systemowej. Na tym etapie wynik fizycznego uruchomienia ma status `NOT_TESTED`.

## Zakresy A, B i C

| Zakres | Kontrakt |
| --- | --- |
| A — preflight V5d1 | `cryptography==50.0.2`, `cffi==2.0.0`, `pycparser==3.0`, `defusedxml==0.7.1`, przypięty CPython/stdlib, dwa główne pliki libvirt oraz `/usr/bin/qemu-img`. |
| B — opcjonalny verifier offline | Odrębny lock i interpreter CP311 albo odrębna kandydatura CP312. Dekoder V3 wymaga przypiętego `zstandard==0.25.0` z backendem `cext`. |
| C — pełny CryptoHunter | Kanoniczna propozycja V5d7 obejmuje 69 wheels i ich pełne ABI. Każdy materiał C zachowuje dotychczasowy format i zakres. |

Do A wchodzą cztery dokładne archiwa z kanonicznej propozycji V5d7, 179
wierszy RECORD, dwa ELF wheels i 27 źródeł z 279 krawędziami AST. Manifest
różnicy obejmuje 65 pozostałych dystrybucji, 11 658 unikalnych ścieżek
docelowych i 718 wpisów ELF. Zależność systemowa `libzstd.so.1` pozostaje w
zakresie `qemu-img`. Moduł Python `zstandard` jest odroczonym importem
dekodera B i pozostaje niedostępny w A.

Rozpatrzenie C nadal obejmuje 12 brakujących nazw PySide6, QPSQL i
`RHPG_9.6`, kontekst RPATH Designera, 18 programów ELF, dostawców Qt/FFmpeg
oraz libpyside/libshiboken. Statyczna kwalifikacja A nie używa tych plików.
Pełne omówienie przypadków pozostaje w [V5d8](stage9_v5d8_physical_abi_evidence.md).

## Builder i przegląd A

`stage9_v5d10_builder.py propose` z osobnego wheelhouse odtwarza wszystkie
RECORD, analizuje METADATA, markery i aktywne zależności przechodnie oraz
odrzuca obce dystrybucje, obcych właścicieli docelowych i dodatkowe ELF.
`build` wymaga świeżego katalogu oraz przekazanego niezależnie hasha dokładnych
bajtów aprobaty. Fizyczny `origins.json` i mapa odtworzona z wierszy V5d7 to
dwa różne artefakty. Należy porównać pochodzenie wybranych archiwów z pierwotnym
plikiem i wykazem. Minimalne wejście buildera zawiera dokładnie cztery rekordy.

Przeznaczenie wiążą osobne identyfikatory w rodzinie
`stage9-v5d10-observer-preflight-*-v1`. Zakres zatwierdzanych pól obejmuje host,
propozycję, manifest native, kod, przegląd dynamiczny, ABI request, ABI
expectations, systemowe bindingi, bazowy runtime i narzędzia hosta. Cztery
pola wyjątków wheel mają puste listy, ponieważ zakres A nie przewiduje
pomijania członków ani współdzielenia plików. Dodatkowy `file_plan_sha256`
wiąże wszystkie ścieżki, tryby i hashe, wraz z venv i launcherem. Kod buildera
sprawdza dokładne identyfikatory, odczytuje piny przed zapisem, porównuje ABI
kopii i publikuje `ENVIRONMENT.json` po końcowym porównaniu drzewa i pinów.
Receipt wiąże aprobatę i zakres. Jego stan `NOT_AUTHORIZED` dotyczy uruchomienia.

W V5d8 pozyskano piny interpretera, `sysconfigdata`, bazowego runtime i
systemowego grafu. Podpisane archiwum V5d9 dostarcza brakujący `patchlevel.h`
CPython 3.12.3. Nagłówek pozostaje w katalogu dowodowym. Statyczny replay A
obejmuje 122 ELF i 263 krawędzie. `libpython3.12.so.1.0` eksportuje
inicjalizatory modułów wbudowanych, lecz nie jest rozszerzeniem importowanym
pod nazwą `libpython3`. Adapter A klasyfikuje tylko ten przypięty systemowy
obiekt w odrębnej roli i zachowuje wymagania wobec rzeczywistych rozszerzeń.
Wejście i zachowanie domyślnego ABI V5d7 pozostają takie same.

Po niezależnej aprobacie dane A obsługuje się poniższymi poleceniami z
zaufanego i uprzednio sprawdzonego checkoutu. `A_APPROVAL_SHA256` i
`A_RECEIPT_SHA256` pochodzą z niezależnego kanału. Każdy pomiar ścieżek
należy wykonać dla dokładnego położenia, które wiąże plan.

```sh
/usr/bin/python3.12 -I -S -B scripts/stage9_v5d10_builder.py propose \
  --repository /path/to/reviewed-source \
  --wheelhouse /path/to/minimal-wheelhouse \
  --origins /path/to/four-reviewed-origins.json

umask 077
/usr/bin/python3.12 -I -S -B scripts/stage9_v5d10_builder.py build \
  --inputs /path/to/inputs.json --approval /path/to/approval.json \
  --approval-sha256 "$A_APPROVAL_SHA256" --output /path/to/fresh-A

/usr/bin/python3.12 -I -S -B scripts/stage9_v5d10_launch.py verify \
  --root /path/to/fresh-A --approval-sha256 "$A_APPROVAL_SHA256" \
  --receipt-sha256 "$A_RECEIPT_SHA256"
```

## Granica uruchomienia preflight

Launcher odmawia dla dodatkowych finderów, obcych korzeni `sys.path`,
nieuzgodnionych wpisów `sys.modules`, podmienionego pakietu i podpakietu spoza
przypiętego korzenia. Kod źródłowy jest odczytywany według hasha z receipt
lub bazowego runtime, z pominięciem cache bytecode. Standardowe moduły
wbudowane i zamrożone pochodzą z zaakceptowanego interpretera. Hook audytu
ogranicza dynamiczne importy, operacje ctypes i wywołanie procesu QEMU do
przypiętego deskryptora oraz argumentów istniejącej ścieżki V5d1.

Kontrola Pythona obejmuje kod i importy w procesie. Ładowanie ELF wymaga
odrębnego widoku systemowego. Operator musi dostarczyć zaufany bootstrap,
prywatne przestrzenie nazw z dokładnym widokiem montowań, readonly root,
prywatny procfs, wyczyszczone deskryptory, AppArmor i seccomp zgodne z osobno
przejrzanymi politykami. Launcher porównuje stan jądra i widoczne drzewo z
autoryzacją. Odrzuca cache loadera, preload, nieprzypięte aliasy oraz
niezakwalifikowane katalogi modułów QEMU i libvirt. Widok wystawia wyłącznie
przypięte pliki i jeden lokalny socket libvirt w trybie odczytowym. Recenzent
potwierdza także tożsamość i zachowanie demona po drugiej stronie socketu,
pin konfiguracji klienta oraz reguły wykonania potomka. Sam status AppArmor
lub seccomp nie potwierdza treści załadowanych reguł.

Autoryzacja `stage9-v5d10-observer-preflight-launch-authorization-v1` wiąże
receipt i aprobatę, hash `mountinfo`, tożsamości przestrzeni nazw, oznaczenie
AppArmor, lokalny socket i jego tożsamość, hash konfiguracji libvirt oraz
aliasy i dozwolone ścieżki danych. Ścieżka żądania ABI prowadzi do dokładnych
bajtów przypiętych w receipt. Launcher porównuje cele aliasów z tym żądaniem
i zatwierdzonym bazowym runtime. Recenzent trzyma dokument poza drzewem A.
To późniejsza operacja. Plan buildera i wynik statyczny nie poświadczają
wyboru dostawcy przez loader ani uruchomienia. Pełny Observer oraz Stage 3B
mają własne granice i autoryzacje.

`STAGE_9=IN_PROGRESS`, `OBSERVER_3A=BLOCKED`, `STAGE_3B=NOT_AUTHORIZED`,
`PRODUCTION=NO-GO`.
