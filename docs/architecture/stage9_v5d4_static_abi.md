# V5d4 — statyczna diagnostyka ABI kontrolera

Narzędzie [`stage9_v5d4_abi.py`](../../scripts/stage9_v5d4_abi.py) czyta pliki
jako dane. Nie uruchamia badanego interpretera, nie importuje `libvirt.py`,
nie ładuje `libvirtmod`, nie wywołuje loadera ani `ldd` i nie wykonuje operacji
libvirt. Uruchamianym narzędziem zaufanego hosta jest wyłącznie stały
`/usr/bin/readelf`; zaufanie do tego narzędzia i hostowego Pythona musi być
ustalone przed diagnostyką.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```

## Wejście i osobne oczekiwania

Żądanie jest obiektem JSON o dokładnie poniższych kluczach. Nieznane klucze,
brakujące pola i powtórzone klucze JSON powodują odmowę CLI. Wszystkie ścieżki
plików są absolutne, znormalizowane i bez dowiązań; systemowe aliasy należy
rozwiązać podczas niezależnego przygotowania danych. Nie przekazuje się
ścieżki do interpretera gościa w celu jego uruchomienia.

| Pole żądania | Znaczenie |
| --- | --- |
| `profile` | Dokładnie `stage9-v5d4-static-abi-v1`. |
| `interpreter` | Rzeczywisty plik ELF przyszłego hostowego interpretera kontrolera. |
| `patchlevel_header` | Przejrzany `patchlevel.h` tego samego wydania CPython; wymagane pojedyncze definicje `PY_MAJOR_VERSION=3`, `PY_MINOR_VERSION=12`, `PY_MICRO_VERSION=3`. |
| `sysconfigdata` | Przejrzane źródło `_sysconfigdata*.py`; AST odczytuje literalne `build_time_vars` bez wykonania pliku. Wymagane `VERSION="3.12"`, `SOABI="cpython-312-x86_64-linux-gnu"`, `MULTIARCH="x86_64-linux-gnu"`, `Py_DEBUG=0`. |
| `libvirt_module` | Kopia `libvirt.py` przeznaczona dla środowiska kontrolera. |
| `libvirt_extension` | Dokładnie plik `libvirtmod.cpython-312-x86_64-linux-gnu.so` przeznaczony dla tego środowiska. |
| `additional_extensions` | Lista wszystkich pozostałych natywnych plików do analizy: rozszerzeń Python, bibliotek `.so`/`.so.N`, pluginów oraz ELF bez rozszerzenia nazwy, np. `QtWebEngineProcess`. Nazwa pola jest wspólna dla tego pełnego wykazu. |
| `import_roots` | Niepusta lista istniejących jawnych katalogów rozwiązywania importów. Narzędzie nie dodaje ich do własnego `sys.path`. |
| `required_imports` | Mapa nazwy modułu na dokładną absolutną ścieżkę jego źródła lub rozszerzenia. Musi zawierać `libvirt` i `libvirtmod` wskazujące powyższe kopie. |
| `library_map` | Mapa nazwy `DT_NEEDED`/absolutnego `PT_INTERP` na dokładny rzeczywisty plik biblioteki lub loadera; każdy napotkany wpis musi istnieć. |
| `pyvenv_cfg` | Absolutna ścieżka do konfiguracji venv z `include-system-site-packages = false` albo `null`, gdy w danym porównaniu venv nie występuje. |

`additional_extensions` obejmuje również ELF dostarczany przez wheel,
który nie jest osiągalny przez `DT_NEEDED` głównego modułu. Builder porównuje
pełny wykaz natywnych członków wheel z tym żądaniem i wynikowymi pomiarami.
Plik o nazwie `.cpython-*` z innym ABI niż wskazane CPython 3.12 jest odrzucany.
Pliki ELF `EXEC` wymagają jawnego `PT_INTERP` z przypiętym loaderem; profil
nie dopuszcza nieopisanego statycznego programu. Obsługiwane są również
wykonywalne pliki PIE typu `DYN`.

Osobny plik oczekiwań ma dokładnie trzy pola:

```json
{
  "profile": "stage9-v5d4-static-expectations-v1",
  "review_reference": "Identyfikator osobnego przeglądu i źródła oczekiwań",
  "sha256": {
    "/absolutna/sciezka/do/konkretnego/pliku": "<dokładne 64 małe znaki hex>"
  }
}
```

Przykład przedstawia strukturę, nie zatwierdzone hashe. Mapa obejmuje każdy
mierzony plik: interpreter, nagłówek, sysconfigdata, konfigurację venv,
`libvirt.py`, rozszerzenia, inicjalizatory pakietów, wskazane importy, wszystkie
przechodnie biblioteki ELF i loader. Brak wpisu lub inny SHA-256 powoduje
odmowę. Wskazanie `review_reference` i zgodność hashy nie uwierzytelniają autora
ani nie ustanawiają niezależnej autoryzacji. Recenzent osobno sprawdza źródło,
wersje pakietów, pochodzenie plików oraz związek nagłówka/sysconfigdata z danym
interpreterem. Sam proces nie potrafi potwierdzić tego pochodzenia.

## Uruchomienie odczytowej diagnostyki

Z niezależnie przejrzanej kopii narzędzi, przez zaufany hostowy Python:

```bash
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  /usr/bin/python3 -I -S -B scripts/stage9_v5d4_abi.py \
  /absolutna/sciezka/abi-request.json \
  --expectations /absolutna/sciezka/abi-expectations.json \
  --expectations-sha256 "$reviewed_abi_expectations_sha256" \
  > abi-stdout.json 2> abi-stderr.log
abi_exit=$?
```

Zmienna `reviewed_abi_expectations_sha256` zawiera hash dokładnych bajtów
oczekiwań, otrzymany niezależnym kanałem przeglądu. Nie należy automatycznie
wyliczać go z kandydata i uznawać za aprobatę. Narzędzie nie generuje oczekiwań
ani podpisów operatora. Zachowuje się oryginalne stdout, stderr, kod zakończenia
i SHA-256 obu raportów; patrz procedura przekazania V5d3/V5d4.

Kod **0** i `STATIC_ABI_MATCH` oznaczają wyłącznie zgodność w opisanym zakresie
statycznym z dostarczonymi oczekiwaniami. Kod **2** oznacza odmowę lub błąd
analizy; błędne argumenty również dają kod 2 przez `argparse`. Niekompletny
wynik, brak raportu lub nienormalny koniec procesu nie stanowią pozytywnego
pomiaru. API dla przejrzanego buildera to
`diagnose(request, expectations) -> dict`; wyjątek `Refusal` zatrzymuje budowę.

## Wykonywane kontrole

Diagnostyka porównuje ELF64/x86_64/little-endian, typ ELF, `PT_INTERP`,
`DT_NEEDED`, `SONAME` i przechodni graf bibliotek. Wersjonowane wymagania
symboli muszą występować w definicjach wskazanej biblioteki. Rozszerzenia
CPython/abi3 muszą eksportować odpowiednie `PyInit_*`; niezdefiniowane symbole
Python muszą być dostępne w grafie interpretera i jego bibliotek, bez polegania
na eksportach przypadkowego innego rozszerzenia.

`RPATH`/`RUNPATH` dopuszcza wyłącznie literalne `$ORIGIN` lub `${ORIGIN}` oraz ich
względne sufiksy ścieżek. Po normalizacji katalog musi istnieć i pozostać pod jawnym
`import_roots`. Puste elementy, inne zmienne, ścieżki globalne, wyjście poza
zatwierdzone katalogi i rozbieżność pierwszego istniejącego kandydata z
`library_map` powodują odmowę. Każdy taki plik nadal wymaga osobnego przypięcia
SHA-256; `$ORIGIN` nie zastępuje przeglądu bibliotek.

Rozwiązywanie importów odrzuca niejednoznaczne moduły, shadowing i brak
inicjalizatorów pakietów. Kontrola dotyczy jawnego `required_imports` oraz
ich inicjalizatorów; nie wyprowadza z tego dowodu zupełności dynamicznej.
W katalogach importów odrzucane są dowiązania, `.pth`, `.pyc`,
`sitecustomize.py`, `usercustomize.py` oraz niezadeklarowane pliki `.so`/`.so.N`.
Każdy znaleziony zadeklarowany natywny plik jest analizowany również wtedy,
gdy nie jest osiągalny od głównego modułu przez `DT_NEEDED`.

Limit pojedynczego pliku wynosi 512 MiB, a wyniku `readelf` 64 MiB; timeout
odczytu jednego ELF to 30 sekund, maksymalnie 1024 ELF w grafie. Przekroczenie
limitu kończy się odmową. Zestaw zawierający większe biblioteki wymaga osobnego
przeglądu limitów, bez obcinania analizy.

## Ograniczenia dowodu i bramka ładowania

Wynik zawiera `measurements` oraz `elf_measurements` jako mapy absolutnych
ścieżek na SHA-256. `elf_edges` pokazuje jawne zależności, a
`python_symbol_sources` wskazuje graf dostarczający symbole Python. To materiał
do porównania przez osobnego recenzenta, nie dowód tożsamości działającego
procesu. Plik może zmienić się po zakończeniu kontroli; snapshot nie ustanawia
blokady lifecycle ani ochrony przed konkurującym zapisującym procesem.

`EXPLICIT_MAP_ONLY_NOT_LOADER_ATTESTATION` nie potwierdza rzeczywistego wyboru
przez loader, `ld.so.cache`, zależności `dlopen`, konfiguracji pluginów,
zgodności instrukcji CPU, wszystkich niewersjonowanych symboli C/C++, ani
skuteczności środowiska izolacji. Statyczny graf bibliotek nie obejmuje
natywnych pluginów pochodzących spoza zadeklarowanego zestawu. Wykazy pakietów,
pochodzenie i pełne natywne inventory wymagają osobnej kontroli narzędziami
propozycji i buildera.

Ładowanie kodu natywnego pozostaje osobnym działaniem wymagającym jawnej
zewnętrznej autoryzacji i uprzednio zatwierdzonego środowiska. Ten interfejs nie
udostępnia flagi ładowania: zawsze zapisuje
`NOT_EXECUTED_REQUIRES_SEPARATE_EXPLICIT_AUTHORIZATION`,
`launch_identity=NOT_VERIFIED` i zachowuje wszystkie blokady Stage 9.
Nie wykonuje testów połączenia libvirt ani lifecycle VM.
