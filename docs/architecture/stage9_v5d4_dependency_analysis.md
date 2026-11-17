# V5d4 — analiza zależności kontrolera

Baza analizy: `913ae53a2e351291f86b05c4e3de1bb56c3559e1`,
repozytorium `Dudzian/Dudzian`. Wnioski bezpieczeństwa sprawdzono w źródłach
V5d1, V5c3, V3 i deklaracjach projektu. Narzędzia Codebase Memory nie są
dostępne w tym środowisku; nie wykonano sprawdzenia pokrycia indeksu.
Wynik AST opisuje statyczny graf, a nie dowód wszystkich możliwych wykonań.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```

## Powtarzalny pomiar źródeł

Przejrzany, zaufany Python hosta może wykonać samodzielny generator
[`stage9_v5d4_closure.py`](../../scripts/stage9_v5d4_closure.py), bez ładowania
żadnego modułu z analizowanego checkoutu:

```bash
/usr/bin/python3 -I -S -B /sciezka/przejrzanych-narzedzi/stage9_v5d4_closure.py \
  --repository /sciezka/checkoutu --output /sciezka/nowego/closure.json
```

Generator czyta AST i `pyproject.toml`, analizuje inicjalizatory pakietów,
rekurencyjne importy lokalne oraz importy wewnątrz funkcji. Dla każdego
źródła zapisuje ścieżkę względną, rozmiar i SHA-256. Lista `static_edges`
podaje źródło, wiersz, moduł, importowane nazwy oraz fazę
`module_initialization` lub `deferred`. Import pod warunkiem na poziomie
modułu nadal należy do fazy inicjalizacji; nazwa fazy nie dowodzi spełnienia
warunku. Domyślne wejście to CLI V5d1. Generator nie wywołuje `find_spec`,
providerów, hooków, kodu pakietów ani `site` badanego środowiska.

Kod 0 oznacza rozwiązanie lokalnych krawędzi i literalnych celów importów
dynamicznych w tym profilu. Kod 2 oznacza brak lokalnego modułu,
nierozwiązany mechanizm dynamiczny albo błąd odczytu. Żaden kod nie przyznaje
akceptacji instalacyjnej: `installation_approved` pozostaje `false`.
Klasyfikacja modułów biblioteki standardowej korzysta z jawnie zapisanej
wersji katalogu zaufanego interpretera narzędziowego. Bajty biblioteki
standardowej docelowego CPython wymagają osobnego wykazu i kontroli ABI.

Generator ma także osobny tryb pełnego, konserwatywnego wykazu materiału
bazowego CPython. Korzenie pochodzą z niezależnie przeglądanej instalacji
hosta i muszą obejmować wszystkie istniejące pozycje `sys.path` startu
`-I -S`, w tym ewentualne archiwum standardowej biblioteki oraz osobny
katalog `lib-dynload`. Nie uruchamia się w tym celu badanego interpretera
gościa. Przykład dla układu, w którym biblioteka i rozszerzenia są pod
jednym katalogiem:

```bash
/usr/bin/python3 -I -S -B /sciezka/przejrzanych-narzedzi/stage9_v5d4_closure.py \
  --base-runtime-root /usr/lib/python3.12 \
  --output /sciezka/nowego/base-runtime.json
```

Opcja `--base-runtime-root` jest powtarzalna; korzeń może być również
zwykłym plikiem, np. `/usr/lib/python312.zip`, jeżeli taki plik rzeczywiście
istnieje. Ten tryb wyklucza `--repository`. API
`base_runtime_inventory(roots: list[Path])` zwraca `files` jako mapę
absolutnych ścieżek do SHA-256 i `native` jako posortowane ścieżki plików
z nagłówkiem ELF. Obejmuje wszystkie zwykłe pliki, **również cache
`__pycache__`, `.pyc` i `.pyo`**. Wykluczone są tylko całe podkatalogi
`site-packages` i `dist-packages`, nieaktywne w przyjętym starcie `-S`;
ich ścieżki są jawnie zapisywane w `excluded`. Plik `.pth` poza tymi
katalogami powoduje odmowę, zamiast milczącego pominięcia kodu startup.

Dystrybucyjny układ CPython może zawierać aliasy plików, np.
`_sysconfigdata`, `sitecustomize.py` lub `config-*/libpython*.so`.
Łańcuch do 16 dowiązań do zwykłego pliku jest zapisywany w `symlinks`
ogniwo po ogniwie, również dla aliasów poza korzeniami. Każdy rekord ma
absolutną ścieżkę aliasu oraz dokładne pola `target` (surowy bezpośredni cel),
`resolved` (końcowy absolutny plik bez dowiązań w przodkach) i `sha256`
końcowego pliku. Typowe dwa ogniwa dystrybucyjnego `libpython*.so`
nie wymagają zmieniania systemowego układu plików.
Cel trafia do `files` i w razie ELF do `native`, **także poza korzeniami**.
Cała struktura aliasów i każde rozszerzenie zakresu wymagają niezależnego
zatwierdzenia; inwentaryzacja nie rozszerza automatycznie zaufania.
Dowiązanie do katalogu, pętla lub przekroczenie limitu 16 ogniw, dowiązanie w przodkach,
nieczytelne dane i pliki specjalne powodują odmowę. Odczyt pliku używa
`O_NOFOLLOW` i porównuje tożsamość oraz metadane przed i po odczycie.
Wykaz nie zastępuje ochrony przed konkurującymi zapisami i późniejszą
zmianą materiału po wykonaniu raportu.

`scope=STATIC_BYTES_SUPERSET_NOT_DYNAMIC_COMPLETENESS` oznacza nadzbiór
bajtów standardowej biblioteki dla osobnego przeglądu. Wykaz nie dowodzi
semantycznej kompletności dynamicznego ładowania, nie wykonuje nawet
`sitecustomize.py` ani cache bytecode i nie przyznaje aprobaty instalacji.
Wszystkie wskazane natywne pliki muszą wejść do odrębnej kontroli ABI.

Dla tej bazy uzyskano **27 źródeł Python i 279 statycznych krawędzi**.
Wykaz źródeł obejmuje:

- inicjalizatory `deployment/__init__.py` i `scripts/__init__.py`;
- CLI `scripts/stage9_observer_kvm_preflight_v5d1.py`;
- `deployment/canonical_json.py`;
- cztery moduły V5d1: `collector_identity`, `kvm_domain`, `kvm_image`, `kvm_preflight`;
- wspólne moduły Observer: `contract`, `offline`, `ubuntu`, `tar_v3`, `package_v3`;
- `apt_v4a`, `apt_v4b`, `gpgv_v4b`, `apt_v4c`, `checkpoint_v4c`, `trust_v4c`;
- `image_v5a`, `member_source_v5b`;
- `pyc_v5c1`, `pyc_execution_v5c3`, `runtime_identity_v5c3`,
  `pyc_reconstruction_v5c2`, `pyc_source_hash_v5c2`, `source_hash_algorithm_v5c2`.

Skrócone nazwy poza `canonical_json` oznaczają pełne ścieżki
`deployment/stage9_observer_<nazwa>.py`. Sześć minimalnych plików z kontraktu
tożsamości V5d1 nie stanowi pełnego importowanego kodu. Oba inicjalizatory
również wykonują kod Pythona i muszą należeć do zatwierdzonego materiału.

## Wykonanie preflight a kontrakt pełnej instalacji

| Zakres | Wymagania i znaczenie |
| --- | --- |
| Import CLI i modułów preflight | Statyczne zewnętrzne korzenie to `cryptography` i `defusedxml`. Pierwszy zapewnia weryfikację podpisów tożsamości; drugi jest wymaganym parserem XML. |
| Odczyt domeny po przejściu wcześniejszych bramek | `LibvirtDomainReaderV5d1.__init__` wykonuje literalny `importlib.import_module("libvirt")`. Binding wymaga zgodnego `libvirtmod` i jego bibliotek ELF. Sam raport brakującej polityki może zakończyć się przed tym importem. |
| Ścieżka Linux obrazu | `fcntl` jest importowany wewnątrz funkcji. QEMU image tooling jest odrębnym natywnym narzędziem, którego bajty i zależności także muszą być przyjęte. |
| Wspólna obsługa archiwów | Importowane moduły V3 zawierają deferred `importlib.import_module("zstandard")`. Zwykłe `collect_preflight_v5d1` nie wywołuje dekompresora; do obsługi tej funkcji potrzebna jest osobna decyzja o profilu i jego zależnościach. |
| Obowiązująca instalacja | Całe `project.dependencies` oraz extra `observer-kvm-preflight`, zgodnie z V5d1. Wynik analizy preflight nie upoważnia do zastąpienia ich minimalnym środowiskiem. |

Kontrakt instalacyjny wymienia **26 wymagań projektu**: `pyarrow`, `protobuf`,
`grpcio`, `PyYAML`, `cryptography`, `psycopg[binary]`, `PyNaCl`, `pydantic`,
`numpy`, `pandas`, `joblib`, `scipy`, `lightgbm`, `ccxt`, `keyring`,
`secretstorage` (Linux), `requests`, `responses`, `httpx`, `anyio`,
`aiosqlite`, `respx`, `jsonschema`, `sqlalchemy[asyncio]`, `PySide6`,
`packaging`; extra dodaje `defusedxml==0.7.1`. Dokładne ograniczenia wersji
są kopiowane bezpośrednio z `pyproject.toml` do wyniku, razem z hashem tego
pliku. `cryptography>=50.0.1,<51` pozostaje wymagane. Extra
`offline-verifier` deklaruje osobno `zstandard==0.25.0`; generator nie
włącza go automatycznie. Typy `types-defusedxml` służą kontroli jakości.

Zbiór zależności przechodnich zależy od faktycznych wersji, extra, markerów
platformy i zawartości dystrybucji. Nie można ustalić jego wersji i hashy
wyłącznie z zakresów `>=` w źródłach. V5d4 wymaga zatem rozwiązania pełnych
deklaracji dla CPython 3.12/Linux x86_64, zebrania wheel METADATA i bajtów
dystrybucji, a następnie odrębnego porównania z niezależnymi oczekiwaniami.
Propozycja wygenerowana lokalnie nie jest źródłem aprobaty dla samej siebie.

## Python, native i startup

Przekazane przez użytkownika obserwacje fizycznego hosta to Ubuntu 24.04.5,
x86_64, CPython 3.12.3, SOABI `cpython-312-x86_64-linux-gnu`, binding
`python3-libvirt` 10.0.0, libvirt 10.0.0 i QEMU 8.2.2. Wskazany plik
`libvirtmod.cpython-312-x86_64-linux-gnu.so` jest przeznaczony dla CPython
3.12. Wcześniejsze `.venv` CPython 3.11 nie spełnia tego ABI przez samą
obecność pliku na dysku ani przez dołączenie katalogu systemowego.
Nie wykonywano ponownego pomiaru tego fizycznego hosta w środowisku prac.

Domknięcie native musi obejmować:

- interpreter, jego loader ELF, bibliotekę standardową i jej rozszerzenia
  (np. `fcntl`, `zlib`, `math`, `_struct`, oraz moduły osiągane przechodnio
  przez `hashlib`, `json`, `subprocess`, `datetime` i parser XML);
- dokładne `libvirt.py`, `libvirtmod` i każdy wymagany moduł towarzyszący
  z przyjętej dystrybucji systemowego bindingu;
- wszystkie natywne elementy wheel, również te nieużywane przy samym
  imporcie preflight, jeśli dystrybucja należy do pełnej instalacji;
- rekurencyjne `DT_NEEDED`, loader, SONAME i zakresy wersji symboli dla
  każdego ELF oraz źródło i SHA-256 plików wybranych przez resolver.

Pełna aplikacja zawiera m.in. rodziny Qt/PySide, Arrow, NumPy/SciPy,
LightGBM, PostgreSQL, libsodium i rozszerzenia kryptograficzne. Ich
rzeczywiste biblioteki mogą być dołączone do wheel lub pobierane z hosta;
same nazwy pakietów nie ustalają kompletnego zestawu SONAME. Statyczna
kontrola ma odczytać te dane z konkretnych ELF. Nie wolno używać `ldd` do
badania niezaufanych plików ani uznać samego suffixu rozszerzenia za dowód
zgodności wersji symboli. Dynamiczne `dlopen` i wybór providerów pozostają
osobną pozycją przeglądu, także gdy `DT_NEEDED` jest domknięte.

Wymagane są jawne, przyjęte roots rozwiązywania importów, kontrola
shadowingu, `pyvenv.cfg`, archiwów `.zip`, cache `.pyc`, `site.py`,
`sitecustomize.py`, `usercustomize.py` i wszystkich `.pth` w zakresach,
które mogłyby uczestniczyć w starcie. Wiersz `import` w `.pth` wykonuje
kod przed wejściem CLI. `-I` nie wyłącza `site`; `-B` nie zabrania odczytu
istniejącego bytecode. Proponowany izolowany start wymaga `-I -S -B`
i osobnego przejrzanego mechanizmu wyboru modułów. Nie wymaga
`--system-site-packages`, globalnej zmiany `sys.path` ani uruchamiania
badanego interpretera gościa. Ocena tych mechanizmów jest odrębna od
niezmienionego kontraktu podpisanej tożsamości V5d1.

## Mechanizmy dynamiczne i granice dowodu

Generator rozpoznaje dokładne mechanizmy `exec`, `eval`, `__import__`,
`importlib.import_module`, aliasy importów, proste przypisania aliasów,
`runpy`, loadery modułów, `ctypes`, `dlopen` i refleksyjny `getattr` na
modułach importujących. `collector.execute(...)` nie jest `exec(...)`,
a `re.compile(...)` nie jest wbudowanym kompilatorem Pythona. Literalny
cel dynamicznego importu jest identyfikowany; nie staje się zatwierdzony.
Cel wyliczany w runtime, kod wykonywany przez `exec`/`eval`, nieznany
loader lub ścieżka natywnej biblioteki blokują statyczne domknięcie.

AST nie rozstrzyga wszystkich przepływów danych, dowolnej refleksji,
zmiany `sys.meta_path`, hooków `.pth`, natywnych providerów ani importów
wewnątrz zewnętrznych dystrybucji. Te ostatnie muszą być analizowane
po pozyskaniu konkretnych wheel i bibliotek systemowych. Wykonanie kodu
natywnego wymaga osobnej jawnej autoryzacji i nie należy do tej analizy.
Testy [`test_stage9_v5d4_closure.py`](../../tests/deployment/test_stage9_v5d4_closure.py)
obejmują brak źródła, inicjalizatory, transitives, aliasy, rzeczywiste
i fałszywe mechanizmy dynamiczne, dowiązania oraz rzeczywisty graf V5d1.
Nie wykonują importów badanych plików, libvirt ani operacji VM.
