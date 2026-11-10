# Stage 9 — odtwarzalność Freshness Runtime i metadane RECORD

Zakres poprawki: builder oparty na rewizji
`44335fe6505aa26452345a7f9e796cecbc9c1b9e`, jego testy i dokumentacja.
Poprawka nie ustanawia zakończenia Stage 9 ani gotowości produkcyjnej.
Wyeksportowane wcześniej kandydaty i ich manifesty pozostają niezmienione.

## Przyczyna różnicy

Pip generuje skrypt `bin/cffi-gen-src` z shebangiem wskazującym interpreter
w katalogu staging. Wpis `../../../bin/cffi-gen-src` w
`cffi-2.1.1.dist-info/RECORD` zawiera hash i rozmiar tego wygenerowanego
skryptu. Dłuższa o sześć znaków nazwa katalogu (`clean-`) wyjaśnia różnicę
207/213 bajtów w dwóch wcześniej zbudowanych kandydatach. Dla bardzo długich
ścieżek pip może również zmienić postać shebanga na skrypt pośredniczący.

Builder usuwał zwykłe pliki z `bin`, w tym skrypty konsolowe i aktywacyjne,
ale pozostawiał ich wpisy w `RECORD`. Były to metadane pliku już
niezainstalowanego, zależne od ścieżki staging. Nie była to wykazana różnica
w pozostałych zainstalowanych bibliotekach Pythona. Porównanie zgłoszone
przez operatora obejmowało 366 obiektów wewnętrznych, z których 365 było
identycznych; nie zastępuje to ponownego testu poprawionego buildera.

Naprawa dotyczy kroku budowania świeżego kandydata. Usunięcie wpisu jest
dopuszczalne wyłącznie dla skryptu rzeczywiście usuniętego w tym kroku.
Brak innego pliku, błędna zawartość lub uszkodzone metadane powodują odmowę.
`RECORD` pozostaje plikiem CSV według
[Recording Installed Projects](https://packaging.python.org/en/latest/specifications/recording-installed-packages/),
z wpisem samego `RECORD` o pustych polach hasha i rozmiaru. Hash całego
`RECORD` nadal należy do manifestu runtime. Wewnętrzna kontrola metadanych
nie zastępuje SHA-256 manifestu, kontroli wheelhouse ani niezależnego podpisu.
Walidator jest przeznaczony dla zamkniętego zbioru wheelów tego buildera:
celowo dopuszcza kanoniczne ścieżki względne i SHA-256 dla podanych hashy.
Nie jest uniwersalnym walidatorem dowolnej instalacji Pythona.

## Domknięcie odmów po przeglądzie PR #3262

Pierwsza poprawka `RECORD` nadal dodawała do planu usuwania każdy zwykły
plik z `bin`. Sam typ pliku nie dowodził jego pochodzenia. Nieznany
`bin/unknown-executable` bez wpisu `RECORD` mógł więc zniknąć przed
zaobserwowaniem błędnej instalacji. Obecnie cały plan jest sprawdzany przed
pierwszym `unlink`, a usuwanie dopuszcza wyłącznie dwa rozłączne źródła:

- Systemowy `venv` Ubuntu 24.04, `with_pip=False`: `activate`, `activate.csh`,
  `activate.fish`, `Activate.ps1`. Te skrypty nie należą do dystrybucji wheel
  i nie mogą być przypisane do jej `RECORD`.
- Zatwierdzone wheels runtime: jedyny skrypt konsolowy to
  `cffi-gen-src = cffi._cffi_gen_src:run`, wskazany przez `entry_points.txt`
  pakietu `cffi==2.1.1`. Wpis musi znajdować się we własnym
  `cffi-2.1.1.dist-info/RECORD`. Żaden runtime wheel nie zawiera `.data/scripts`.
  Skrypty `pip`/`pip3` należą do narzędzia buildowego i nie są instalowane.

Nazwy aktywacji potwierdzono przez utworzenie świeżego środowiska systemowym
Pythonem w jednorazowym kontenerze. Właścicieli skryptów sprawdzono odczytowo
w archiwach wheelhouse, po weryfikacji SHA-256. Nieznany zwykły plik,
dopisanie obcego skryptu do poprawnego CSV, przejęcie `cffi-gen-src` przez
inny pakiet albo przypisanie aktywacji venv do `RECORD` powodują odmowę
bez usunięcia istniejących skryptów i bez zmiany metadanych.

Drugi problem wynikał z walidacji plików CSV bez walidacji dokładnego zbioru
dystrybucji. Siódma poprawna dystrybucja lub duplikat pod inną nazwą katalogu
mogły mieć rozłączne, poprawne wpisy i przejść tę kontrolę. Sonda dodatkowo
tworzyła słownik przez comprehension, co mogło nadpisać wcześniejszą
dystrybucję o tej samej nazwie i ukryć nadmiar.

`_installed_records` wymaga obecnie dokładnie sześciu nazw i wersji z części
runtime zatwierdzonego katalogu wheelhouse. Każdy `METADATA` musi mieć
pojedyncze pola `Name` i `Version`, poprawną gramatykę nazwy i własny wpis
w `RECORD`. Normalizacja PEP 503 łączy ciągi `-`, `_` i `.` oraz ignoruje
wielkość liter. Duplikat jest sprawdzany przed dodaniem do zbioru; katalog
musi mieć kanoniczną nazwę instalacji wheel. `_PROBE` również sprawdza
powtórzenie przed przypisaniem wartości w słowniku.

Przy okazji wykryto powiązany wariant: skan `*.dist-info` pomijał nazwy
z wielkimi literami, a samo pokrycie plików mogło zostać zapewnione przez
`RECORD` innego pakietu. Tak mogły ukryć się metadane `.DIST-INFO` lub
`.egg-info`, rozpoznawane przez `importlib.metadata`. Dwa testy odtworzyły
brak odmowy; skan uwzględnia teraz wielkość liter niezależnie od zapisu,
a metadane `egg-info` są odrzucane jako niedozwolone w zamkniętej instalacji
wheel. Usunięto ten wariant w zakresie tego samego PR.

Dodano 25 przypadków do testów `RECORD` (łącznie 59), w tym porównanie
pełnej zawartości instalacji przed i po odmowie oraz rzeczywisty proces
sondy z powtórzoną dystrybucją. Natywny test odtwarzalności dodatkowo wymaga
jawnie `len(files) == 366` dla każdego z obu kandydatów.

## Trzy niezależne twierdzenia

| Twierdzenie | Wymagany dowód | Granica wniosku |
| --- | --- | --- |
| Odtwarzalność na identycznej bazie | Dwa buildy offline z identycznych wheelów i lockfile, na tym samym systemie, z identycznymi wersjami i bajtami Pythona oraz zależności natywnych, w staging o różnych nazwach i długościach; równość pełnego inwentarza wewnętrznego i hashy plików, w tym `RECORD` | Dotyczy porównanej bazy i wejść; nie dowodzi identyczności buildów na innym wydaniu pakietu Ubuntu |
| Zgodność funkcjonalna między wydaniami Ubuntu | Oddzielny audyt integralności oraz testy rzeczywistych importów, Ed25519 i libpq dla każdego wydania pakietów | Sukces testów na `python3.12=3.12.3-1ubuntu0.16` oraz `.17` nie oznacza identycznych bajtów interpretera ani bibliotek systemowych |
| Integralność konkretnego kandydata | Weryfikacja podpisu Ed25519 i pełnego manifestu tego kandydata, SHA-256 plików wewnętrznych i zewnętrznych, typów, trybów i dowiązań | Manifest innego kandydata nie jest zamiennikiem; zgodność ABI i numer `3.12.3` nie dowodzą tożsamości binarnej |

Porównanie odtwarzalności analizuje osobno `files`, `external_files`,
`interpreter`, wejścia i pola tożsamości. `environment_id` oraz wynikające
z niego `installation_path` mogą celowo się różnić. Porównanie hashy całych
manifestów bez uwzględnienia tych pól nie jest testem odtwarzalności plików.
Wersje pakietów systemowych oraz podpisane hashe zewnętrzne nie są takimi
polami do pominięcia: zmiana tych materiałów oznacza inną bazę porównania.

## Inwentarz systemowy i bytecode

`_external_inventory` zbiera systemowy interpreter, stdlib z wyłączeniem
`site-packages`/`dist-packages`, konfigurację loadera, opcjonalne archiwum
`python312.zip` i domknięcie zależności natywnych. Zachowuje literalne
dowiązania oraz wszystkie wymagane cele i katalogi nadrzędne. Skan native
obejmuje rozszerzenia stdlib i pliki `.so`/`.so.*` z kandydata; rekursywne
rozwiązywanie zależności nadal odmawia przy brakującej bibliotece.

Całe stdlib stanowi konserwatywny nadzbiór materiałów potrzebnych do jednej
sondy importów. Pliki konfiguracyjne używane do budowania rozszerzeń czy
cache modułów niewczytywanych w konkretnej próbie mogą być w tej próbie
zbędne. Sama obserwacja jednej listy importów nie dowodzi jednak, że
przyszła ścieżka usługi ich nie użyje. Ta poprawka nie ogranicza inwentarza.

`PYTHONDONTWRITEBYTECODE=1`, `--no-compile` i `-B` ograniczają tworzenie nowych
plików `.pyc`; nie zabraniają odczytu ani wykonywania już istniejącego
bytecode. `-I -s` izoluje ścieżki i user-site, ale nie zmienia tej zasady.
CPython może wykonać cache w `__pycache__`, jeżeli zgadza się timestamp
i rozmiar źródła lub właściwy hash. Domyślna polityka dopuszcza także
unchecked-hash `.pyc`, które nie porównuje zawartości źródła. Import
sourceless obsługuje starszy układ `module.pyc` bez `module.py`.
To wystarcza, aby zbiorcze pominięcie `.pyc` osłabiło kontrolę integralności
potencjalnie wykonywanych bajtów.

Zgłoszone 150 dodatkowych plików `.pyc` i 422 zmienione cache spośród
486 zmienionych obiektów zewnętrznych mogą wynikać ze stanu cache hostów;
sam bilans nie dowodzi ich pochodzenia ani niewykonywalności. Pozostałe
64 zmienione obiekty obejmują rzeczywiste materiały systemowe i nie mogą
być uznane za identyczne z powodu ABI lub numeru wydania Pythona.

Ewentualne przyszłe ograniczenie wymaga osobnego kontraktu: albo wymuszenia
braku odczytu niezainwentaryzowanego bytecode przed startem interpretera,
albo ścisłej, egzekwowanej listy materiałów importowanych przez usługę.
Nie wystarcza dodanie `-B`, usunięcie `.pyc` z listy, pominięcie katalogów
`__pycache__` czy uzyskanie równych hashy po filtracji. Zmiana musiałaby
zachować wszystkie wykonywane warianty bytecode, biblioteki natywne i cele
dowiązań. Nie jest częścią tego PR.

## Walidacja i kontrakt bezpieczeństwa

Testy importów bytecode wykonują oddzielny proces `-I -s -B` dla cache
timestamp, unchecked-hash i modułu sourceless. Test inwentarza sprawdza
obecność i SHA-256 `.pyc` oraz zachowanie wszystkich korzeni native,
zewnętrznej biblioteki i jej literalnego dowiązania. Testy buildera sprawdzają
odtwarzalność w różnych staging oraz odmowy uszkodzonych metadanych, braków
plików i manipulacji wheelhouse. Rzeczywista próba wymaga systemowego
CPython 3.12 na jednorazowym Ubuntu 24.04; pominięty test nie jest dowodem
pozytywnego wyniku. Wyniki konkretnego wykonania przedstawiono poniżej i w raporcie PR.

Pozostają kontrole fail-closed, dokładne SHA-256 wejść i manifestu,
pełny inwentarz native, zakaz wykonywalnych skryptów instalacyjnych
w runtime i niezależny mechanizm podpisywania Ed25519. Ten zestaw testów
nie publikuje runtime, nie uruchamia usługi produkcyjnej i nie dopuszcza
środowiska do produkcji.

## Wyniki wykonania — 10 listopada 2026

Końcowy zestaw pięciu modułów testowych wykonał **240 testów bez błędów
ani pominięć** (23,77 s) w jednorazowym kontenerze Ubuntu 24.04.5,
z systemowym `python3.12=3.12.3-1ubuntu0.17`, jako UID 1000 i przy
`--network none`. Źródła i wejściowy wheelhouse zamontowano tylko do odczytu.
Narzędzia testowe miały oddzielny venv; zależności runtime zainstalowano
z tego samego lockfile przez `--no-index --require-hashes --only-binary=:all:`.
Nie uruchamiano próby publikującej `freshness_runtime_noble_trial.py`.

| Sprawdzenie | Wynik |
| --- | --- |
| Dwie pary buildów: zwykły wheelhouse oraz tryby 0500/0400 | Obie pary zgodne przy staging długości 65 i 234 znaków; ścieżki interpretera miały 80 i 249 znaków |
| Pełny inwentarz wewnętrzny każdej pary | 366 identycznych obiektów: 313 plików, 49 katalogów i 4 dowiązania; wszystkie SHA-256 plików identyczne |
| Metadane instalacji | Wszystkie 6 plików `RECORD` obecne w manifeście; niezależna kontrola CSV, istnienia, hashy, rozmiarów i kompletności plików przechodzi |
| Inwentarz zewnętrzny każdej pary | 1345 identycznych obiektów, w tym 573 pliki `.pyc`; bez filtrowania cache i zależności natywnych |
| Porównanie manifestów | Wszystkie pola równe poza celowo różnymi `environment_id` i `installation_path` |
| Negatywne próby `RECORD` | Odmowy błędnego CSV/UTF-8, SHA-256, rozmiarów, duplikatów, braku wpisu własnego, ścieżek poza kontraktem, brakujących plików/skryptów/RECORD, nieujętych plików i dowiązań |
| Negatywne próby wheelhouse | Odmowy zmiany bajtu bez zmiany rozmiaru, brakującego i dodatkowego wheel; brak utworzonego staging i manifestu |
| Istniejące zabezpieczenia | Testy podpisów, integralności, izolacji importów, Ed25519, libpq i zakazu nadpisania przechodzą |
| Przenośny zestaw na Debianie | 230 zaliczonych; 10 wymagających Ubuntu pominięto, po czym wykonano je w powyższym kontenerze |
| Bramki jakości | Ruff Tier A, jawny format zmienionych modułów, obowiązkowy `python -m pre_commit run ruff-format --all-files`, skonfigurowany hook mypy i dodatkowa kontrola buildera dla Windows przechodzą |
| Betterleaks 1.9.0 | Istniejący hook zmian i pełny skan czystej kopii śledzonych źródeł z niezmienionym baseline: brak wykryć |

Podział końcowej regresji: runtime 80, metadane `RECORD` 59, bytecode 4,
aplikacja 51, builder 46. Wszystkie próby natywne rzeczywiście wykonano,
włącznie z wariantami wheelhouse zwykłego oraz 0500/0400.

Przebiegi robocze tej aktualizacji również ujawniły i wyjaśniły błędy:
cztery skupione regresje przed poprawką potwierdziły brak odmowy dla obcego
pliku, dodatkowej dystrybucji i dwóch duplikatów. Pierwsza wersja nowej
implementacji miała kolizję lokalnej nazwy zmiennej mapy wersji z oczekiwanym
hashem: mypy zgłosił jeden błąd, a testy dały 20 niezaliczonych, 70 zaliczonych
i 10 pominiętych. Po usunięciu kolizji uzyskano 90 zaliczonych i 10
pominiętych, a po trzech dodatkowych regresjach 93 zaliczone i 10 pominiętych.
Dwie dalsze próby `.DIST-INFO`/`.egg-info` początkowo nie zgłaszały odmowy;
po poprawieniu skanu przechodzą. Końcowe wyniki powyżej obejmują wszystkie
te przypadki. Dwie pełne próby natywne po aktualizacji dały po 240 poprawnych
wyników (24,50 s oraz końcowe 23,77 s). Po pierwszej doprecyzowano test sondy:
wymaga konkretnego komunikatu odmowy duplikatu i pustego stdout, aby inna,
wcześniejsza odmowa nie dawała fałszywie pozytywnego wyniku. Przy globalnym
runnerze ten jednostkowy test modeluje tylko warunek różnej bazy venv;
rzeczywistą izolację nadal sprawdzają natywne buildy. Test ukrytej `.egg-info`
używa `PKG-INFO` i potwierdza obecność siódmej dystrybucji przez rzeczywisty
`importlib.metadata` przed odmową buildera. Niezależny zestaw `RECORD` oraz
bytecode po tych zmianach dał 63 poprawne wyniki.

Nie pozostał nierozwiązany błąd bezpieczeństwa wykryty w tej aktualizacji;
wynik nie zastępuje niezależnego audytu ani kwalifikacji hosta.

Identyfikacja bazy testowej:

- Bazowy obraz Ubuntu: `sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852`.
- Lokalny obraz z narzędziami: `sha256:50b49eca557bc83d7f915fb6f1471023f59a848defa969c4e764db3510e26555`.
- SHA-256 zachowanego zestawienia dpkg: `f7695f098ff4c214119842f8e77b12469e287a867cf1ab6e15ed07ef1f49c77b`.
- `libc6=2.39-0ubuntu8.9`, `libssl3t64=3.0.13-0ubuntu3.16`.
- Lockfile: `d8e1d874671b0db936058d38d0094179140c7669ae30f78ebbde89cfebc2acb3`.
- `cffi-2.1.1.dist-info/RECORD` po poprawce w obu parach:
  `411644d77f30393754d3b9fc79531adbc6d0c00cfa8819fce90e4b253df38e85`.

Pierwszy przebieg wykrył trzy błędy przygotowania danych nowych testów
bytecode: wcześniejsze testy bootstrapu pozostawiały
`sys.pycache_prefix='/dev/null'`. Dodano lokalną izolację cache w nowych
testach; ponowny pełny zestaw pierwotnej rewizji dał 215 poprawnych wyników.
Po dodatkowych poprawkach przeglądu pełna regresja dała 240 poprawnych wyników.
Rzeczywiste buildy i porównania odtwarzalności przeszły także w pierwszej
próbie. Nie zmieniano mechanizmu bootstrapu ani wygenerowanych kandydatów.

Pełny skan zastanego workspace zgłosił również ignorowany, nieśledzony
`tests/__pycache__/test_runtime_bootstrap.cpython-311-pytest-9.1.1.pyc`
z 6 października, zawierający testowy PEM ze źródła ujętego w baseline.
Ten cache nie jest częścią PR; skan zmian i czystych źródeł przeszedł.
Nie zmieniano baseline ani reguł skanera.
Przy ponownym skanie kopii źródeł argument absolutny spowodował 18
powtórnych zgłoszeń znanych fixtures, ponieważ prefiks katalogu zmienił
odciski ścieżek. Wszystkie 18 odcisków po usunięciu prefiksu odpowiadało
istniejącemu baseline. Powtórzenie utrzymywanego polecenia z `dir .`
w katalogu kopii źródeł dało kod 0 i brak nowych wykryć. Skan objął 4155
śledzonych plików, około 67,29 MB; baseline pozostał bez zmian.

Wynik kontenera nie stanowi kwalifikacji fizycznego hosta. Nowego buildera
nie uruchamiano tu na pakiecie `.16`; wcześniejsze obserwacje operatora
dla `.16` i `.17` pozostają odrębnym dowodem zgodności funkcjonalnej.

## Działania po mergowaniu i pozostałe ryzyka

1. Na ustalonej bazie Ubuntu 24.04 zamrozić dokładne wersje pakietów,
   wheelhouse i lockfile; wykonać dwa świeże buildy z różnych staging.
2. Zachować wyniki testów, pełne inwentarze i identyfikację bazy. Sprawdzić
   brak nieaktualnych wpisów `RECORD` i równość wszystkich obiektów
   wewnętrznych. Nie poprawiać dawnych eksportów ani ich manifestów.
3. Dla każdej odmiennej bazy systemowej przeprowadzić osobne testy
   funkcjonalne i integralności. Nie wymagać identycznych hashy bibliotek
   z różnych wersji pakietów Ubuntu.
4. Nowych kandydatów przekazać do niezależnego audytu i ewentualnej odrębnej
   ceremonii podpisania. Merge PR nie zastępuje zatwierdzenia, publikacji,
   kwalifikacji hosta ani dalszych bramek Stage 9.

Systemowy cache nadal może zmieniać stan zewnętrznego inwentarza między
buildami. Aktualizacja interpretera, biblioteki, konfiguracji loadera albo
wykonywalnego cache po zebraniu manifestu nadal wymaga odmowy przy kontroli
danego kandydata i nowej, niezależnej oceny materiałów. Odtwarzalność dwóch
buildów nie stanowi dowodu bezpieczeństwa bazy systemowej ani kompletnej
kwalifikacji produktu.
