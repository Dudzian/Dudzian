# Stage 9 V5d6 — inwentarz odmów greenlet i skrypt MASM

Zbadano wszystkie 99 plików i 7 wpisów katalogów w oryginalnym
`greenlet-3.5.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`.
Archiwum ma 621404 bajty i SHA-256
`975736b002ed080d124cf81a79cb7e05cb26d6b3f5c7a7b651c0fcce70353aa1`.
CRC oraz pełny `RECORD` zweryfikowano na oryginalnych bajtach. `RECORD`
zawiera 98 poprawnych hashy i rozmiarów oraz własny wpis z pustymi polami.
Inwentarz przypisuje każdy plik do jednej kategorii i zachowuje tożsamości
plików szczególnych. [Wynik inwentaryzacji](evidence/stage9_v5d6_greenlet_full_inventory.json) ·
[Hashe wszystkich plików](evidence/stage9_v5d6_greenlet_inspection.json)

| Wynik reguł badanych z samych bajtów wheel | Pliki |
| --- | ---: |
| Mapowanie i reguły członka bez dodatkowej stwierdzonej odmowy | 91 |
| Przypięty kandydat pominięcia nagłówka `.data/headers` | 1 |
| Identyczne bajty nagłówka pod ścieżką mapowaną do runtime | 1 |
| Odmowa formatu wykonywalnego `.cmd` | 1 |
| Rozpoznane biblioteki ELF, osobne oczekiwania native/ABI | 3 |
| Obiekty COFF dla Windows, poza klasyfikacją native obecnej reguły | 2 |

## Dwie stwierdzone blokady

1. Niezmieniony `inspect_wheel_bytes()` zwraca
   `WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED` dla
   `greenlet/platform/setup_switch_x64_masm.cmd`. Po potwierdzeniu tej odmowy
   sprawdzono osobno tę samą regułę dla wszystkich wpisów plikowych, również
   leżących po `.cmd` w kolejności ZIP. Tylko ten plik ją narusza.
2. `greenlet/greenlet.h` ma bajt w bajt tę samą zawartość co
   `greenlet-3.5.6.data/headers/greenlet.h`. Oba pliki mają 4755 bajtów,
   tryb `0100644` i SHA-256
   `b33e69611490a9e7603add80320c4b65d4e33bea9caff24cbe0884251f48009f`.
   Inspektor przypina drugi wpis jako osobnego kandydata pominięcia. Pierwszy
   wpis trafia jednak do mapowania `greenlet/greenlet.h`. Wywołanie
   `header_inventory()` z tym mapowaniem i rzeczywistym kandydatem daje
   `HEADER_RUNTIME_INCLUSION_FORBIDDEN`. To wynik izolowanego sprawdzenia
   późniejszej reguły. Normalna analiza wheel odmawia wcześniej na `.cmd`.

Mapowanie odtworzono w pamięci z oryginalnych bajtów, z zachowaniem `.cmd`.
Nie generowano propozycji ani aprobaty. Nie wykryto kolizji ścieżek wewnątrz
wheel, innych plików z nieobsługiwanym formatem, obcych cache PYC, plików
startowych, nieprawidłowych tagów SOABI ani relokacji native z `.data`.
Porównanie z innymi wheels i rozwiązanie zależności całego zestawu pozostaje
`NOT_TESTED_IN_CLOUD`.

## `setup_switch_x64_masm.cmd` jako plik używany podczas budowania

| Pole | Zmierzona wartość |
| --- | --- |
| Ścieżka | `greenlet/platform/setup_switch_x64_masm.cmd` |
| SHA-256 | `6690a5509794d2e8c43d24d63527a93f45b67fd5e2610280f10292a15a2ef045` |
| Rozmiar i format | 143 bajty ASCII, dwie linie z CRLF |
| Typ i tryb ZIP | Zwykły plik, `0100644` (33188), bez bitów wykonywalności Unix |
| CRC-32 | `0c16b380` |
| Wpis `RECORD` | `greenlet/platform/setup_switch_x64_masm.cmd,sha256=ZpClUJeU0ujEPSTWNSepP0W2f9XiYQKA8QKSoVou8EU,143` |

Pierwsza linia wywołuje `vcvarsall.bat` z Visual Studio 9.0 dla `amd64`.
Druga wywołuje asembler `ml64` w trybie `/c`, ze źródłem
`switch_x64_masm.asm` i plikiem wyjściowym `switch_x64_masm.obj`. Obie
ścieżki znajdują się obok `.cmd` w tym samym oryginalnym archiwum. Kod
`switch_x64_masm.asm` deklaruje `slp_switch PROC FRAME` oraz wywołania
`slp_save_state_asm` i `slp_restore_state_asm`. Wiersz 37 nagłówka
`switch_x64_msvc.h` wskazuje zewnętrzny plik asemblera MASM.
`slp_platformselect.h` wybiera ten nagłówek dla Windows x64 MSVC lub MinGW64,
a dla GNU na amd64 bez x32 wybiera `switch_amd64_unix.h`.

`greenlet/platform/__init__.py` jest pusty. `greenlet/__init__.py` importuje
symbole z rozszerzenia `_greenlet`. Dosłowny ciąg `setup_switch_x64_masm`
występuje w członkach wheel wyłącznie w `RECORD`. To wspiera rozpoznanie
skryptu jako narzędzia budowania dla Windows x64. Sam odczyt archiwum nie
dowodzi braku wszystkich możliwych odwołań w docelowym środowisku.
Brak bitów wykonywalności Unix nie wyklucza wykonania przez interpreter
poleceń Windows. Obecna reguła nadal odmawia dla `.cmd`. Pominięcie skryptu
ani ogólny wyjątek dla skryptów nie mają aprobaty.

## Pozostałe artefakty i granice pomiaru

`switch_x64_masm.obj` i `switch_arm64_masm.obj` mają odpowiednio nagłówki
COFF z kodami maszyn `0x8664` (AMD64) i `0xaa64` (ARM64). Obiekty zawierają
tablice sekcji i symboli, bez opcjonalnego nagłówka obrazu i flagi obrazu
wykonywalnego. Obecna klasyfikacja inspektora nie rozpoznaje tych członków
jako native i ich format nie wyzwala istniejącej odmowy. Wymagają odrębnego
przeglądu przed jakąkolwiek decyzją o dopuszczeniu archiwum.

Trzy `.so` mają nagłówki ELF64 x86-64 little endian typu `DYN`, pasujące
tagi SOABI i oczekiwane eksporty `PyInit_*`. Żaden nie deklaruje
`RPATH` ani `RUNPATH`. `readelf` odczytał tablice z bajtów w pamięci.
W dowodzie zapisano `DT_NEEDED`, wersje symboli i niewywiązane symbole
Pythona. Rozwiązanie tych zależności i potwierdzenie eksportów wymaga
docelowych CPython, libvirt, biblioteki systemowej, mapy ścieżek
i niezależnych przypięć ABI. Bez nich status to `NOT_TESTED`, a brak lub
rozbieżność oczekiwań nadal powoduje odmowę. Kod z wheel pozostał
niewykonany, a archiwum i `RECORD` zachowały oryginalne bajty.

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```
