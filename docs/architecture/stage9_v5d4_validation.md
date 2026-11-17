# V5d4 — raport walidacji

Baza: `913ae53a2e351291f86b05c4e3de1bb56c3559e1`. Walidacja wykonana
17 listopada 2026 w izolowanym, nieuprzywilejowanym środowisku Cloud z
CPython 3.12.14. Nie jest to pomiar fizycznego hosta Ubuntu operatora ani
kwalifikacja CPython 3.12.3/libvirt 10.0.0/QEMU 8.2.2.

Surowe raporty są w [katalogu dowodów](evidence/stage9_v5d4_validation/).
Każda końcowa kontrola ma osobne stdout, stderr, plik `.exit` oraz
`.command.json` z dokładnym argv. Zachowano JUnit, XML pokrycia, JSON
Semgrep/Betterleaks i wcześniejszy surowy raport regresji V5d2. Plik
`SHA256SUMS` wiąże dokładne bajty tych raportów; nie stanowi niezależnej aprobaty.
Oryginalne `semgrep.stderr` i `semgrep-rules.log` zachowano w bezstratnym gzip,
ponieważ wyjście narzędzia zawiera końcowe spacje. Ich treści nie poprawiano;
`RAW_SHA256SUMS` zawiera hashe strumieni po dekompresji.

| Kontrola | Wynik i zakres |
| --- | --- |
| V5d1 + V5d3 + V5d4 | 805 passed, 1 skipped; pominięty jest istniejący test natywnej odmowy na hoście innym niż Linux. |
| V5d4 | 162 testy: ABI 46, closure/base runtime 36, zależności 45, staging/launcher/odczyty 35. |
| V5d2 | 63 passed, bez provisioningu hosta; zachowany oryginalny log, XML i exit. |
| Pokrycie pięciu narzędzi V5d4 | Około 88% z pomiarem gałęzi; dokładne wartości i brakujące wiersze w `regression.stdout` i `coverage.xml`. |
| Ruff | Cały dotychczasowy scope `bot_core core scripts tests deploy stage6_samples ui`, kod 0. |
| Format | Dokładny `python -m pre_commit run ruff-format --all-files`, po dodaniu nowych plików do Git, kod 0 i brak zmian formattera. |
| mypy | Skonfigurowane zakresy Linux i Win32, bez argumentów nazw plików, 424 pliki, kod 0. |
| Semgrep | Niezmienione blokujące reguły dla całego istniejącego zakresu, zero ustaleń; fixtures 5/5. |
| Betterleaks | Pełny working tree z dotychczasową bazą i redakcją treści, zero wycieków. |

Testy pozytywne przygotowania używają syntetycznych bajtów i zastępują
granicę hosta/ELF, pozostawiając rzeczywisty odczyt, porównania i zapis
nowego katalogu. Moduły-fixtures celowo zawierają kod zgłaszający wyjątek
przy wykonaniu: staging go nie wykonuje. Test bramki native zastępuje
wyłącznie wywołanie importu rejestratorem; nie importuje libvirt. Regresje
V5d1 zachowują istniejące syntetyczne klucze/testowe polityki, nie tworzą
podpisów operatora ani niezależnego dopuszczenia.

Negatywne scenariusze obejmują brak zależności przechodniej, niewłaściwą
wersję/marker/extra, niedopasowany SHA-256 dystrybucji i członka RECORD,
niezatwierdzone natywne bajty, niezgodny SOABI/Python ELF, brak symbolu,
brak biblioteki i wersji symbolu, błędną konfigurację importów, `.pth`,
shadowing, unsafe RUNPATH, brak base runtime/cache evidence, zmianę
pliku podczas odczytu, niekompletny materiał przekazania i brak jawnej
bramki. Odczyt zmieniający tylko atime nie jest fałszywie uznawany za
modyfikację treści; zmiany rozmiaru, inode, uprawnień, mtime/ctime pozostają
kontrolowane. Nieoczekiwany FIFO w runtime odmawia przed otwarciem.

Odczytowy smoke generatora bazowego wykazu na lokalnym `/usr/lib/python3.13`
objął 1281 plików, 49 ELF i cztery ogniwa dowiązań; nie wykonywał tego
interpretera. To dodatkowe sprawdzenie techniczne generatora, bez dowodu
zgodności docelowej wersji 3.12.3. Codebase Memory było niedostępne;
analizę oparto na rzeczywistych źródłach, bez twierdzenia o pokryciu indeksu.

Nie instalowano do `/opt/cryptohunter` ani `/etc/cryptohunter`, nie używano
sudo, nie importowano docelowego libvirt/libvirtmod, nie wykonywano metod
libvirt, nie uruchamiano VM, badanego interpretera gościa ani bytecode,
nie zmieniano obrazów i nie przyznawano autoryzacji. Weryfikatory/testy
korzystają z zaufanego runtime narzędziowego, w tym testowej cryptography;
nie jest to autoryzowane ładowanie badanego zestawu natywnego.

Pozostają: niezależny wybór i akceptacja pełnego wheelhouse, pochodzenia
systemowych pakietów/runtime, rzeczywiste ABI/loader/dlopen na docelowym
hoście, chroniony launch i immutable view, podpisana polityka V5d1,
osobna domena/obrazy, kwalifikacja i odrębna autoryzacja eksperymentu.
Nowe narzędzia odmawiają przy braku tych wejść; nie tworzą zastępczej aprobaty.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```
