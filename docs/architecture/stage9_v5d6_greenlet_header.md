# Stage 9 V5d6 — oddzielny kontrakt pominięcia nagłówka greenlet

Baza: aktualny zdalny `main` sprawdzony przed pracą,
`ad85f19a15b6b1acffd9146a8b08760a3208ed7a`. Przeczytano `AGENTS.md`.
Codebase Memory MCP ani jego CLI nie były dostępne; wnioski sprawdzono
bezpośrednio w źródłach i testach. Ta zmiana jest kandydatem draft PR.

## Przyczyna i dokładna tożsamość

`materialized_path()` V5d4 dopuszcza wyłącznie `.data/purelib` i
`.data/platlib`. Nagłówek C API należy do poprawnej kategorii Wheel `headers`,
ale nie ma celu w przygotowanym środowisku importów. Wcześniejsza odmowa
`UNSUPPORTED_WHEEL_DATA` była więc zgodna z dotychczasowym kontraktem.

Oryginalne archiwum pobrano z adresu PyPI wskazanego w
[pełnym dowodzie statycznym](evidence/stage9_v5d6_greenlet_inspection.json).
Nie przepakowano go, nie zmieniono jego `RECORD`, nie instalowano i nie
wykonano jego kodu. Archiwum pozostaje poza Git.

| Pole | Wynik pomiaru |
| --- | --- |
| Dystrybucja / wersja | `greenlet` / `3.5.6` |
| Wheel | `greenlet-3.5.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl` |
| SHA-256 wheel | `975736b002ed080d124cf81a79cb7e05cb26d6b3f5c7a7b651c0fcce70353aa1` |
| Rozmiar wheel | 621404 bajty |
| Członek | `greenlet-3.5.6.data/headers/greenlet.h` |
| SHA-256 nagłówka | `b33e69611490a9e7603add80320c4b65d4e33bea9caff24cbe0884251f48009f` |
| Rozmiar / tryb / typ | 4755 bajtów / `0100644` (33188) / regular |
| SHA-256 oryginalnego `RECORD` | `5971269a77a3a3465d9f9e23a9a20841ce2a9d771658e5d57f65be2f0953a016` |
| Integralność ZIP | PASS; CRC wszystkich 99 plików; 7 wpisów katalogów |
| `RECORD` | Dokładnie 99 wpisów; 98 poprawnych SHA-256 i rozmiarów; własny wpis pusty |
| Nieobsługiwane pliki `.data` | Dokładnie jeden, wskazany nagłówek |

Oryginalny wpis `RECORD`:

```text
greenlet-3.5.6.data/headers/greenlet.h,sha256=sz5pYRSQqedgOt2AMgxLZdTjO-qcr_JMvgiEJR9IAJ8,4755
```

Statyczny inspektor `scripts/stage9_v5d6_inspect_greenlet.py` odtwarza pełny
inwentarz bez ekstrakcji. Plik raportu tworzy wyłącznie pod nieużytą ścieżką,
co chroni oryginalny wheel przed nadpisaniem. Potwierdza tekst nagłówka, brak ELF i bitów
wykonywalności. Trzy członki ELF pozostają w osobnym zakresie native/ABI.

## Polityka i odrębna aprobata

Moduł `scripts/stage9_v5d6_headers.py` definiuje politykę
`stage9-v5d6-greenlet-3.5.6-non-runtime-header-v1`.
Inspektor zależności najpierw weryfikuje pełny oryginalny `RECORD` oraz
wszystkie bajty członków, a dopiero potem klasyfikuje nagłówek. Sprawdza
dystrybucję, wersję, nazwę i SHA-256 całego wheel oraz dokładny zbiór
nagłówków. Usunięcie nagłówka i ponowne zapisanie `RECORD` również odmawia.

Osobny `inspect_header()` wymaga dokładnej ścieżki, SHA-256, rozmiaru i trybu
zwykłego pliku. Odrzuca symlinki, bity wykonywalne, ELF, PE, Mach-O i shebang.
Nie rozszerzono `materialized_path()` ani surowego kopiowania ZIP o `headers`.
Nieznane nagłówki, dodatkowy nagłówek, błędne `.data`, `scripts`, `data`
i pozostałe kategorie nadal odmawiają.

Kandydat trafia do osobnego `package.header_omissions`, a pełen plan zawiera
osobno mapowania `materialized_members` i inwentarz `header_omissions`.
`omitted_members` zachowuje wyłącznie dotychczasowe pominięcia PYC.
Polityka oraz pomiary nie tworzą aprobaty.

Weryfikacja przyjmuje osobny argument `approved_headers` i CLI
`--approved-headers`. Kontrakt oczekiwań ma dokładnie pięć kluczy:

- `policy`: wskazany identyfikator polityki;
- `installation_contract_exception`: `OMIT_PINNED_NON_RUNTIME_HEADER`;
- `members`: dokładny pojedynczy rekord dystrybucji, wheel i nagłówka,
  obejmujący SHA-256, rozmiar, tryb, typ, politykę i `runtime_inclusion=EXCLUDED`;
- `review_reference`: niepusta referencja do niezależnego przeglądu;
- `decision`: niezależna decyzja `APPROVED_FOR_HEADER_OMISSION`.

Brak lub niespójność kontraktu oznacza odmowę. Nie dostarczono ani nie
wygenerowano rzeczywistej aprobaty. Testy używają wyraźnych deklaracji
`TEST ONLY` i zastępują piny wyłącznie w prywatnych fixture.
Uwierzytelnienie recenzenta pozostaje poza tym lokalnym kontraktem spójności.

Niezależne pozostają aprobaty dwóch pominięć PYC, 62 współdzielonych plików
PySide6, nagłówka greenlet oraz bibliotek natywnych i ABI. Żadna z nich nie
zastępuje pozostałych ani nie autoryzuje wykonania.

## Builder, receipt i launcher

Builder przyjmuje osobne `approval.wheel_header_omissions`. Pełny plan
zachowuje niezmienne bajty archiwów i odtwarza inspekcję oraz porównanie planu
przed pierwszym zapisem. Zmiana bajtów, tożsamości, deklaracji pominięcia
lub dodanie nagłówka do mapowania runtime powoduje odmowę. Dotychczasowe
kontrole `O_NOFOLLOW`, `O_EXCL`, tożsamości katalogów i kolizji pozostają.

Receipt zachowuje oryginalne tożsamości wheel, `wheel_materialization`
z osobnymi `header_omissions`, oraz pola:

- `header_omissions`: dokładny inwentarz z tożsamością dystrybucji, SHA-256
  oryginalnego wheel i oryginalną ścieżką, SHA-256, rozmiarem, trybem i typem nagłówka;
- `header_omission_expectations`: dokładny osobny kontrakt recenzenta;
- `header_omission_policy`: identyfikator polityki;
- `header_runtime_inclusion`: `EXCLUDED`.

Kopia `header-audit.py` jest objęta pełnym inwentarzem receipt. Launcher po
kontroli drzewa sprawdza ten kontrakt, zgodność z `dependency_verification`,
oryginalną tożsamość archiwum i brak nagłówka w runtime, również pod aliasem
lub przez symlink. Brak metadanych przy deklarowanym greenlet również odmawia.
Zewnętrzny digest receipt i chroniony, niezmienny widok środowiska pozostają
odrębnymi wymaganiami. Dotychczasowa jawna bramka native load jest niezmieniona.

## Zgodność schematów i migracja propozycji

| Dokument | Zgodność z formatem sprzed V5d6 |
| --- | --- |
| Proposal (`stage9-v5d4-dependency-proposal-v1` lub `stage9-v5d5-dependency-proposal-v1`) | Identyfikatory profili pozostały te same. V5d6 dodaje `header_omissions` na poziomie propozycji i każdego pakietu, również przy pustym zbiorze nagłówków. Starszy dokument z poprawnym dawnym SHA-256 różni się od propozycji odtworzonej z tych samych wejść. Przy udanej inspekcji archiwów weryfikator zgłasza `PROPOSAL_CONTENT_MISMATCH`. Nowa propozycja ze starym zatwierdzonym SHA-256 kończy się `PROPOSAL_SHA256_MISMATCH`. |
| Dependency verification (`stage9-v5d4-dependency-verification-v1` lub `stage9-v5d5-dependency-verification-v1`) | Wynik V5d6 zawiera `header_omissions`, `header_expectations` i `header_policy`. Przy pustym zbiorze inwentarz ma status `NO_HEADER_OMISSIONS`, a oczekiwania mogą mieć wartość `null`. Wynik wylicza weryfikator. Builder nie przyjmuje wcześniejszego raportu jako zastępstwa weryfikacji propozycji. |
| Receipt (`stage9-v5d5-prepared-environment-v1`) | Przy braku kandydata, pól kontraktu nagłówka i `header-audit.py` launcher akceptuje starszy format w części dotyczącej nagłówka. Nadal weryfikuje drzewo, kontrakt współdzielenia oraz pozostałe dowody. Receipt deklarujący greenlet 3.5.6 lub pominięcie nagłówka wymaga całego kontraktu V5d6 i zgodnej sekcji `dependency_verification`. Starszy receipt bez tego kontraktu lub z dawnym zagnieżdżonym wynikiem weryfikacji zostaje odrzucony. |

Do nowej weryfikacji lub budowy należy wygenerować propozycję ponownie
narzędziem `stage9_v5d4_dependencies.py propose` z aktualnej rewizji V5d6,
używając oryginalnych archiwów, `pyproject.toml` i odpowiadającego
`origins.json`. Dla profilu współdzielenia zachowuje się flagę
`--allow-shared-candidates`. Odtworzenie jest możliwe dopiero po przejściu
statycznych bramek archiwów. Odmowa inspekcji zatrzymuje procedurę.

Niezależny recenzent ocenia nową propozycję i zatwierdza nowy SHA-256 jej
kanonicznego JSON. Zatwierdzony digest jest używany w `verify --expected-sha256`
oraz `approval.proposal_sha256` buildera. Ręczne dopisanie pustych pól albo
wyliczenie nowego digestu przez operatora nie zastępuje tej aprobaty.
Odrębne oczekiwania PYC, współdzielenia, nagłówka oraz native/ABI nadal
podlegają swoim kontrolom. Pominięcie nagłówka wymaga `--approved-headers`
w weryfikacji i `approval.wheel_header_omissions` w builderze. Receipt dla
nowej budowy musi pochodzić z tej budowy, a jego digest i chroniony widok
runtime podlegają osobnej weryfikacji.

## Następna rzeczywista odmowa — bez obejścia

Inspekcja rzeczywistych bajtów potwierdziła kandydata nagłówka V5d6, ale
pełny `inspect_wheel()` nadal kończy się exit 2:

```text
REASON=WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED
MEMBER=greenlet/platform/setup_switch_x64_masm.cmd
SHA256=6690a5509794d2e8c43d24d63527a93f45b67fd5e2610280f10292a15a2ef045
SIZE=143
MODE=33188
```

To dołączony skrypt Windows z poleceniem wywołania `vcvarsall.bat`. Nie został
wykonany. V5d6 nie nadaje wyjątku plikom `.cmd`, nie pomija go i nie osłabia
bramki wykonywalnych formatów. [Dowód następnej odmowy](evidence/stage9_v5d6_next_refusal.json)
zachowuje dokładny hash archiwum i członka. Osobny test regresyjny wymaga
tej odmowy także wtedy, gdy syntetyczny nagłówek spełnia wszystkie swoje piny.

Rozszerzony przegląd wszystkich 99 plików potwierdził także odrębną,
późniejszą odmowę `HEADER_RUNTIME_INCLUSION_FORBIDDEN`. Mapowany
`greenlet/greenlet.h` ma bajty identyczne z nagłówkiem przypiętym do
pominięcia. Kontrolę odtworzono na mapie z rzeczywistych bajtów bez
obchodzenia odmowy `.cmd`. Inwentarz opisuje też dwa obiekty COFF i trzy
biblioteki ELF oraz statyczne dowody, że `.cmd` służy budowie obiektu MASM
dla Windows x64. [Inwentarz i ocena skryptu](stage9_v5d6_greenlet_build_member.md).

## Pełny wheelhouse — do walidacji na laptopie

Operator potwierdził, że oryginalny zestaw znajduje się wyłącznie na fizycznym
Ubuntu 24.04.5, CPython 3.12.3, Linux x86_64:

```text
/home/dudzian/stage9-v5d4-wheelhouse-candidate-NTUwn8B7/wheelhouse/
/home/dudzian/stage9-v5d4-wheelhouse-candidate-NTUwn8B7/origins.json
```

69 wheels, 416633346 bajtów i wcześniejsze 69/69 porównań SHA-256 z PyPI są
informacją operatora. Cloud nie ma tych rzeczywistych bajtów; pełnego zestawu
nie oznaczono jako zweryfikowanego ani nie odtworzono go z nowych wersji.
Do repozytorium nie dodano wheels. Pełna kontrola po scaleniu poprawki
pozostaje zadaniem operatora; merge nie został wykonany przez agenta.

Poniższe polecenie uruchom z checkoutu zatwierdzonej rewizji repozytorium na
laptopie. Zapisuje pełny SHA, wersję interpretera, raport JSON, stderr i kod
wyjścia w osobnym katalogu w katalogu domowym. Ścieżki wejściowe pozostają
w oryginalnym katalogu operatora.

```bash
(
  set -eu
  repo=$(git rev-parse --show-toplevel)
  input=/home/dudzian/stage9-v5d4-wheelhouse-candidate-NTUwn8B7
  umask 077
  report=$(mktemp -d /home/dudzian/stage9-v5d6-wheelhouse-static.XXXXXXXX)
  git -C "$repo" rev-parse HEAD > "$report/head.sha"
  /usr/bin/python3 --version > "$report/python-version.txt" 2>&1
  validator_exit=0
  /usr/bin/python3 -I -S -B "$repo/scripts/stage9_v5d6_inspect_wheelhouse.py" \
    --pyproject "$repo/pyproject.toml" \
    --wheelhouse "$input/wheelhouse" \
    --origins "$input/origins.json" \
    > "$report/wheelhouse.json" 2> "$report/stderr.log" || validator_exit=$?
  printf 'VALIDATOR_EXIT=%s\n' "$validator_exit" > "$report/exit.txt"
  printf 'REPORT_DIR=%s\n' "$report"
  cat "$report/exit.txt"
  exit "$validator_exit"
)
```

Narzędzie odczytuje oryginalne wejścia, zachowuje SHA-256 archiwów, wyniki
poszczególnych inspekcji i kody odmów. Sumy kandydatów są liczone z faktycznie
sprawdzonych danych. Ocenę pełnego domknięcia uruchamia jedynie po przejściu
kontroli statycznych i liczników. Odmowa, w tym zachowana `.cmd`, pozostaje
w raporcie z kodem wyjścia 2. Cały pomiar jest statyczny i nie tworzy aprobat.

W JSON v2 `expected_counts` opisuje oczekiwany zestaw. W `measurements`
każda wartość ma status `MEASURED` dla kompletnego zakresu danego licznika,
`PARTIAL` dla zmierzonego podzbioru lub `NOT_TESTED` z `value: null`, gdy
brakuje podstawy pomiaru. `wheels_evaluated` podaje rozmiar tej podstawy.
`archive_scan` rozdziela odczyty archiwów, ich sumę bajtów i ukończone inspekcje.
Przy późniejszej odmowie `archives[].observations` zachowuje kandydatów
zmierzonych na zaliczonych etapach. Inwentarz współdzielenia podaje zakres
sprawdzonych map. `closure_status` oddziela sprawdzenie domknięcia od samych
pomiarów. Liczba kandydatów może być znana jednocześnie ze statusem `REFUSED`.
[Przykład diagnostyki ograniczonej do oryginalnego greenlet](evidence/stage9_v5d6_greenlet_diagnostic_v2.json).

Oczekiwania dla oryginalnego zestawu 69/62/2/1:

```text
WHEELS_CHECKED=69
EXPECTED_SHARED_PATHS=62
EXPECTED_FOREIGN_PYC_OMISSIONS=2
EXPECTED_HEADER_OMISSIONS=1
```

Wyniki testów i ograniczenia narzędzi zapisano w
[raporcie walidacji](stage9_v5d6_validation.md). Testy i statyczne pomiary
nie nadają kwalifikacji fizycznej, aprobat native/ABI ani zgody produkcyjnej.

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```
