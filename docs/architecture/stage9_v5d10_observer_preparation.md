# Stage 9 V5d10 — przygotowanie Observera

Baza dowodowa to `52a18d6c423a3e4b2e24c05933518fe0c7bd5352` (PR #3297).
Decyzja dla profilu preflight pozostaje **WSTRZYMAĆ MATERIALIZACJĘ** do czasu
uzyskania pełnego pliku wejść oraz odrębnej aprobaty dla tego zakresu.
Kandydaci opisani poniżej są materiałem do przeglądu. Builder i launcher
obsługują dotychczasowe kontrakty wykonawcze.

## Trzy zakresy

| Zakres | Identyfikator materiału do przeglądu | Wymagane wejścia | Decyzja |
| --- | --- | --- | --- |
| A. Observer preflight V5d1 | `stage9-v5d10-observer-preflight-proposal-v1` | Domknięcie V5d1, `cryptography` z zależnościami, `defusedxml`, bazowy CPython, binding potrzebny do odczytu libvirt, `qemu-img` i ich systemowe domknięcie ABI | Odrębny minimalny wheelhouse ma mniejszą powierzchnię kwalifikacji. Przed dopuszczeniem wymagane są przypięte pliki hosta i egzekwowana granica importu i ładowania. |
| B. Opcjonalny offline verifier | `stage9-v5d10-offline-deb-inventory-wheel-candidate-v1` | Odrębny dekoder i interpreter, z przypiętym `zstandard` oraz własnym domknięciem ABI dla konkretnego wejścia i platformy | Przygotować osobną aprobatę wyłącznie dla operacji, które używają dekodera. Istniejący lock CP311 dotyczy CP311. |
| C. Pełny runtime aplikacji | `stage9-v5d10-full-runtime-review-v1` | Kanoniczna propozycja V5d7, 69 wheels i pełny manifest ABI, zachowane kontrakty współdzielenia i pominięć | Wstrzymać do rozstrzygnięcia pozostałych krawędzi i kontekstów ładowania. |

Domknięcie V5d1 obejmuje 27 plików i 279 krawędzi AST. Statyczne korzenie zewnętrzne
obejmują `cryptography` i `defusedxml`. Dwa odroczone cele literalne to
`libvirt` w ścieżce odczytowego preflight oraz `zstandard` w dekoderze
archiwów Ubuntu. Rozliczenie A wymaga jawnego punktu wejściowego, który nie
udostępnia dekodera, oraz testu odmowy dostępu do niego także przez już
zaimportowane moduły. `-I -S -B`, brak wielkiego wheelhouse na ścieżkach
importu i weryfikacja korzeni są konieczne. Docelowa kontrola musi także
objąć importy podpakietów, zawartość `sys.modules`, chroniony obraz plików
oraz listę dozwolonych dostawców i konfiguracji natywnych dla mechanizmów `dlopen`.
Osobno sprawdza się, że proces `qemu-img` ma przypiętą ścieżkę i zamknięte
środowisko oraz że binding nie może przejść do odmiennego backendu.
Przyszłe wykonanie Observera ma osobne wymagania tożsamości uruchomienia,
izolacji, polityki oraz autoryzacji. Wynik V5d1 sam ich nie spełnia.

## Kontrakt CPython

Odtworzono podpisany łańcuch Ubuntu dla dwóch plików z wersji
`3.12.3-1ubuntu0.17` na `amd64`. Istniejąca kontrola `_python_contract`
potwierdziła wersję `3.12.3`, `SOABI=cpython-312-x86_64-linux-gnu`,
`MULTIARCH=x86_64-linux-gnu` i `Py_DEBUG=0` przez analizę bajtów jako danych.

| Wejście | Pakiet archiwalny | Bajty | SHA-256 |
| --- | --- | ---: | --- |
| `patchlevel.h` w pozasystemowym katalogu dowodowym V5d9 | `libpython3.12-dev` | 1299 | `f2f4800f05108bef77f6164484aba437b86b81e34744cbb5378f7bb2ccc69414` |
| `/usr/lib/python3.12/_sysconfigdata__x86_64-linux-gnu.py` | `libpython3.12-minimal` | 49505 | `ee5bb5493fa4575f7788e1b0e8a067b9c463f9705dcb36f5eb571d1c0812ea82` |

W ABI request ścieżka `patchlevel_header` wskazuje istniejący plik z dowodu
V5d9. Ścieżka `sysconfigdata` wskazuje zmierzony plik hosta. Dwa przypięcia
są częścią przyszłej mapy ABI razem z dokładnym interpreterem i pozostałymi
pomiarami. Sama zgodność powyższych dwóch plików rozstrzyga wyłącznie kontrakt
ich wspólnych pól.

## Granica w istniejącym builderze i launcherze

Weryfikator zależności odbudowuje pełną propozycję z autorytatywnego
`pyproject.toml`, wheelhouse i originów. Wymaga dokładnego zbioru natywnych
członków i pełnego ABI ścieżek przed wydaniem receipt. Zmiana etykiety lub
zmniejszenie planu wejściowego kończy się odmową. Test obejmuje również
wyzerowanie manifestu native przy pełnej propozycji.

Launcher dopuszcza ścieżkę zgodności ze starszym formatem wyłącznie dla
historycznego `stage9-v5d4-prepared-environment-v1`. Nieznany lub brakujący
identyfikator formatu kończy walidację przed odczytem pomocnika runtime i
przed dodaniem findera. Regresja używa poprawnego hasha syntetycznej projekcji
receipt i sprawdza, że kandydat A dochodzi wyłącznie do odmowy. Rzeczywisty
receipt V5d5/V5d7 nadal przechodzi własne kontrole.

## Punkt wznowienia

1. Udostępnić oryginalne surowe archiwa V5d7 i V5d8 oraz oryginalny wheelhouse
   z originami. Sprawdzić SHA-256 archiwum V5d7
   `8c51cdbe12613b42ba7bc8de56eb9f5d19b9fb19f4e9800e6715d148bcb31b8e`
   i kanonicznej propozycji V5d7
   `7f5f6e7f885a368ccdbb844b11534ad7d09871eeb2d8b45484a6bf8fa95aacc4`.
   Porównać brakujące piny z istniejącymi wynikami, przed zleceniem nowych pomiarów.
2. Zbudować kompletny, odrębny spis A. Przypiąć wejścia do świeżego,
   nieistniejącego katalogu poza drzewem systemowym i związać identyfikatory
   profilu, planu plików, ABI i receipt. Sprawdzić syntetycznie, że profile B/C,
   obce biblioteki, podmienione pliki oraz przejścia poza zakres A powodują odmowę.
3. Przekazać dokładne bajty i skróty do niezależnego przeglądu A. Przy odmowie
   sprawa wraca do wskazanego wejścia. Wyłącznie odrębna autoryzacja operatora
   pozwala później na materializację danego profilu. Native load, odczyt libvirt
   i uruchomienie Observera mają własne dalsze bramki.

`STAGE_9=IN_PROGRESS`, `OBSERVER_3A=BLOCKED`, `STAGE_3B=NOT_AUTHORIZED`,
`PRODUCTION=NO-GO`.
