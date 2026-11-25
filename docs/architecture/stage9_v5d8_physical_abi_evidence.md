# Stage 9 V5d8 — uzupełnienie fizycznych dowodów ABI

V5d8 kontynuuje fizyczny pomiar V5d7 wykonany na rewizji
`e097a4de25fedb09ba47706914dfc332b8a1f03e`. Kanoniczna propozycja fizyczna ma
SHA-256 `7f5f6e7f885a368ccdbb844b11534ad7d09871eeb2d8b45484a6bf8fa95aacc4`.
Obejmuje 720 ścieżek ELF, w tym 704 unikalne hashe. Weryfikacja fizycznego
wejścia korzysta z dokładnych bajtów archiwum i propozycji.

## Braki wykazane przez V5d7

- Zmierzony korpus hosta urywa się na 47 nazwach `DT_NEEDED`. W domknięciu
  interpretera i podstawowego powiązania libvirt otwartych pozostaje 18 nazw.
  Odczyt cache, rzeczywistych aliasów i zależności tranzytywnych ma uzupełnić
  te krawędzie oraz domknięcie `qemu-img` i rozszerzeń bazowego CPython.
- Spośród zależności ELF wheel 13 nazw PySide6 nie miało kandydata w
  wheelhouse ani fizycznej próbce cache. Ubuntu Noble dostarcza m.in. nazwę
  `libbz2.so.1` przez pakiet `libbz2-1.0`. Trzeba zmierzyć fizyczny alias i
  SONAME docelowego pliku. Plugin QPSQL wymaga dodatkowo wersji `RHPG_9.6`,
  której zmierzona systemowa `libpq.so.5.16` nie deklaruje.
- `PySide6/Qt/plugins/designer/libPySidePlugin.so` ma `DT_RPATH`
  `$ORIGIN/:$ORIGIN/Qt/lib:$ORIGIN/../shiboken6/` i nie ma `DT_RUNPATH`.
  Dwa z tych katalogów nie istnieją w projektowanym drzewie. Graf uwzględnia
  `RPATH` w domknięciu tranzytywnym i oddzielnie traktuje `RUNPATH`, używany
  dla bezpośrednich zależności obiektu deklarującego ten znacznik. Program
  `PySide6/designer` ma własny `DT_RUNPATH`, w którym znajduje trzy biblioteki
  Qt wymagane także przez plugin. Ich dostępność w procesie przy ładowaniu
  pluginu wymaga osobnego dowodu. Builder zapisuje pliki wheel w trybie
  `0600`, obejmując również 18 programów ELF. Profil uruchamiający te programy
  potrzebuje ustalonej polityki trybów.

## Jeden przebieg odczytowy

Pobrać skrypt z przejrzanej rewizji V5d8. `--repository` wskazuje czyste
drzewo zatwierdzonego commita, a `--physical-v5d7` oryginalne fizyczne
archiwum o SHA-256
`8c51cdbe12613b42ba7bc8de56eb9f5d19b9fb19f4e9800e6715d148bcb31b8e`.
Użyć nowej ścieżki archiwum poza repozytorium. Na Ubuntu 24.04 operator
uruchamia jako zwykły użytkownik:

```sh
/usr/bin/python3.12 -I -S -B /path/to/stage9_v5d8_collect_physical.py \
  --repository /path/to/approved-checkout \
  --physical-v5d7 /path/to/stage9-v5d7-physical-evidence.tar.gz \
  --output /path/to/stage9-v5d8-physical-static-evidence.tar.gz
```

Archiwum zawiera dokładny HEAD, `SHA256SUMS`, stderr i kody zakończenia,
closure AST i bazowy runtime V5d4, dane CPython, tożsamości plików systemowych,
graf kandydatów oraz odczyty dynamicznych symboli i relokacji. Kolektor
porównuje stary i bieżący stan jako kontrolę aktualności. Przebieg częściowy
pozostawia dowody i kod błędu. Przed przeglądem operator porównuje skrót
archiwum oraz weryfikuje `SHA256SUMS` w rozpakowanym katalogu.

## Warunki następnego etapu

Zaktualizować fizyczny graf i jawny plan ABI po otrzymaniu archiwum.
Rozstrzygnąć ścieżki wymagane dla Observer, odrębnie rozliczyć komponenty
opcjonalne i ustalić pliki oczekiwań w niezależnym przeglądzie. Ocena pełnego
wheelhouse według obecnego kontraktu V5d4 obejmuje wszystkie zadeklarowane ELF.
Statyczny graf kandydatów nie potwierdza kolejności działania rzeczywistego
loadera. Autoryzacja native load ma własną bramkę.

Lokalne testy i macierz CI korzystają z danych syntetycznych. Kryterium 70%
pokrycia zmienionego kodu obejmuje dane kolektora z Linux Ubuntu w istniejącym
raporcie łącznym. `STAGE_9=IN_PROGRESS`, `OBSERVER_3A=BLOCKED`,
`STAGE_3B=NOT_AUTHORIZED`, `PRODUCTION=NO-GO`.
