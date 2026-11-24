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
inwentarz bez ekstrakcji. Potwierdza tekst nagłówka, brak ELF i bitów
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

Po uzyskaniu zatwierdzonej rewizji można zachować surowy raport poleceniem:

```bash
python3 -I -S -B scripts/stage9_v5d6_inspect_wheelhouse.py \
  --pyproject pyproject.toml \
  --wheelhouse /home/dudzian/stage9-v5d4-wheelhouse-candidate-NTUwn8B7/wheelhouse/ \
  --origins /home/dudzian/stage9-v5d4-wheelhouse-candidate-NTUwn8B7/origins.json \
  > /home/dudzian/stage9-v5d6-wheelhouse-static.json
```

Narzędzie odczytuje oryginalne wejścia, zachowuje kody odmów oraz SHA-256
archiwów i nie generuje aprobat. Wymaga liczników 69/62/2/1 i pełnego domknięcia
zależności. Kolejna odmowa, w tym zachowana `.cmd`, kończy je exit 2.
Oczekiwane liczniki nie są wynikiem wykonania w Cloud:

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
