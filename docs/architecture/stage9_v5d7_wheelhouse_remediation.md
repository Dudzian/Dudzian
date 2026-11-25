# Stage 9 V5d7 — remediacja wheelhouse

Baza V5d6 to `7e6c21ae8e2472aaad6b6946a9c8545083f1f559`. Archiwum dowodowe SHA-256 `9f57fc7e34adf94c7cdf734e406e8066df0fd909871011110a479d8add0a1bc2` oraz wszystkie pięć wewnętrznych sum zweryfikowano. Raport fizyczny rejestrował 62 pełne inspekcje, siedem odmów archiwów i `NOT_TESTED` dla domknięcia zależności. Podawał 62 kolizje z trzema właścicielami, a ówczesny kontrakt mógł sprawdzić tylko przypiętą parę.

W Cloud pobrano z PyPI 69 kopii o dokładnie tych samych rozmiarach i SHA-256 co w raporcie fizycznym. Łączny rozmiar to 416 633 346 bajtów. Manifest źródeł tego pomiaru został utworzony odrębnie z potwierdzonych adresów PyPI. Procedura operatora korzysta z fizycznych plików i oryginalnego `origins.json`. V5d7 sprawdził wszystkie 11 971 wierszy `RECORD` i zmierzył domknięcie 69 dystrybucji, 27 wymagań wejściowych i 124 krawędzi dla docelowego CPython 3.12.3.

## Kontrakty

- **Greenlet 3.5.6.** Oba nagłówki są identyczne i deklarują C API dla kompilacji downstream. Gotowe rozszerzenia ELF mają eksporty `PyInit` i systemowe `DT_NEEDED`. Skrypt `.cmd` ustawia narzędzia Visual Studio i wywołuje `ml64`, a dwa obiekty są wejściami COFF dla Windows ARM64 i x86-64. V5d6 dalej rozlicza nagłówek `.data/headers/greenlet.h`. V5d7 pomija cztery pozostałe przypięte pliki po pełnym `RECORD`, dokładnym sprawdzeniu tożsamości i osobnej aprobacie. Wyłączenie obu kopii nagłówka odbiera możliwość kompilacji downstream przy użyciu tego drzewa runtime.
- **SciPy 1.18.1.** Wszystkie trzy `extending.pyx` zaczynają się od `#!/usr/bin/env python3`, po którym następują dyrektywy Cython i składnia `cimport`/`cdef`. Każdy plik jest wejściem `meson.build` dla testu `test_extending.py` w odpowiednim podpakiecie `linalg`, `optimize` lub `special`. Zidentyfikowany zakres utraty funkcjonalności obejmuje te przykładowe kompilacje testowe. Kontrakt przypina trzy dokładne pliki wraz z rozmiarem, trybem, typem i tożsamością archiwum.
- **Core Metadata.** `more_itertools` 11.2.1, `pydantic` 2.14.3, `python_dateutil` 4.0.0, `typing_inspection` 0.4.5 i `urllib3` 2.8.0 mają wersję 2.5. Ponowna pełna inspekcja ich archiwów przeszła bez dalszych odmów. `more_itertools` deklaruje `Import-Name: more_itertools`, a pozostałe cztery nie deklarują nazw importu. Walidator V5d7 sprawdza składnię, niejednoznaczność i nakładanie się wyłącznych deklaracji między dystrybucjami. Prywatność nazwy nie wyłącza kontroli konfliktu. Zależności nadal wynikają z `Requires-Dist`, `Requires-Python`, `Provides-Extra` i markerów celu. Obsługiwana lista kończy się na 2.5. Semantyka `Dynamic` w sdist z wersji 2.6 pozostaje poza tym profilem. [PyPA Core Metadata](https://packaging.python.org/en/latest/specifications/core-metadata/) · [PEP 794](https://peps.python.org/pep-0794/) · [PEP 808](https://peps.python.org/pep-0808/)
- **PySide6 6.10.3.** Każda z 62 ścieżek ma dokładnie trzech właścicieli (`PySide6`, `PySide6-Addons`, `PySide6-Essentials`). Zweryfikowano rzeczywistą ścieżkę ZIP, dokładną ścieżkę docelową, SHA-256, rozmiar, pełny tryb Unix i typ każdego członka. Odrębna aprobata musi pokryć cały nowy kontrakt i potwierdzić przegląd `PySide6/__init__.py`, `PySide6/_config.py` i `PySide6/_git_pyside_version.py`. Dwa wcześniejsze pominięcia PYC mają osobną, zachowaną politykę.

## Migracja i bramki

Profil propozycji wybiera `propose --v5d7`. Weryfikator ustala profil z propozycji powiązanej z zewnętrzną sumą SHA-256 i odtwarza ją z archiwów. Wymaga niezależnych oczekiwań dla exact native inventory, PYC, nagłówka V5d6, współdzielenia V5d7 i siedmiu nowych pominięć. CLI weryfikatora używa `--approved-artifacts`, a wejście aprobaty buildera używa `wheel_artifact_omissions`. Dawne oczekiwanie współdzielenia dotyczy wyłącznie pary V5d5. W profilu V5d7 trzeba zatwierdzić nową propozycję i trzyosobowy zakres. Wcześniejsze oczekiwania PYC i V5d6 mogą odpowiadać wyłącznie temu samemu, dokładnemu zakresowi.

Builder weryfikuje nową propozycję, zachowuje niezmienne kopie bajtów ZIP, jeszcze raz sprawdza `RECORD` i manifest, zapisuje wspólny plik jeden raz oraz sprawdza rzeczywiste drzewo przed receipt. Nowy receipt wiąże weryfikację z kodem polityki współdzielenia V5d7, jej przypiętą bazą V5d5 oraz audytem pominięć. Launcher sprawdza niezależną sumę receipt, pełne drzewo i zgodność danych weryfikacji z receipt, w tym brak pominiętych bajtów pod aliasami. Stare receipt nadal podlegają własnym kontraktom. Diagnostyka rejestruje dowody i kandydatów. Niezależne oczekiwania, ABI, zgoda na native load i kwalifikacja hosta pozostają osobnymi bramkami.

## Pomiar formatów

Na kopiach zgodnych z fizycznym SHA-256 zinwentaryzowano 835 członków ze zidentyfikowanym formatem lub przesłanką wykonania. Wszystkie 720 ELF to ELF64 x86-64. Mają 704 unikalne skróty i obejmują 416 bibliotek współdzielonych (w tym 210 pluginów Qt), 286 rozszerzeń Python oraz 18 programów. Wśród pozostałych wpisów są dwa obiekty COFF i dwa archiwa statyczne z dziewięcioma osadzonymi ELF. Liczby 507 kandydatów rozszerzeń według nazwy i 797 członków z bitami wykonywalności opisują nakładające się zbiory. Maksymalne zmierzone wymagania wersji symboli to `GLIBC_2.34`, `GLIBCXX_3.4.29` i `CXXABI_1.3.13`. Porównanie żądanych nazw `DT_NEEDED` z nazwami plików i `SONAME` wskazało 188 nazw z kandydatem w wheelhouse i 96 bez takiego kandydata.

## Jeden przebieg operatora

W katalogu lokalnego klonu repozytorium na laptopie wklej cały blok. Pobiera wskazany commit i uruchamia kolektor w osobnym, czystym worktree. Skrypt sprawdza SHA źródeł i wymaga `/usr/bin/python3.12` w wersji CPython 3.12.3. Używa oryginalnego `wheelhouse/` i `origins.json` z `/home/dudzian/stage9-v5d4-wheelhouse-candidate-NTUwn8B7` oraz raportu `/home/dudzian/stage9-v5d6-audit-PFvqXhAF/stage9-v5d6-full-evidence.tar.gz`.

```bash
(
  set -euo pipefail
  source_sha=aebb00d0212dbaf14dec260aa6cf26a68e63b555
  repo=$(git rev-parse --show-toplevel)
  output_parent=$(cd -P -- "${TMPDIR:-/tmp}" && pwd -P)
  source_dir=$(mktemp -d "$output_parent/stage9-v5d7-source.XXXXXXXX")
  trap 'git -C "$repo" worktree remove -- "$source_dir" 2>/dev/null || rmdir -- "$source_dir"' EXIT
  git -C "$repo" fetch --no-tags --no-write-fetch-head origin "$source_sha"
  git -c core.hooksPath=/dev/null -C "$repo" worktree add --detach -- "$source_dir" "$source_sha"
  /bin/bash "$source_dir/scripts/stage9_v5d7_collect_physical.sh" "$source_sha" "$output_parent"
)
```

Skrypt wypisuje `ARCHIVE`, `SHA256_FILE` i `COLLECTOR_EXIT`. W prywatnym katalogu wynikowym pozostawia jedno archiwum `stage9-v5d7-physical-evidence.tar.gz` i plik z jego sumą SHA-256. Sprawdza sumę zewnętrzną, rozpakowuje utworzone archiwum w prywatnym katalogu tymczasowym i weryfikuje jego wewnętrzny `SHA256SUMS`. Archiwum zawiera pomiar, stderr, kod zakończenia, SHA źródeł, ścieżki wejściowe i kanoniczną propozycję V5d7, jeżeli kolektor mógł ją utworzyć. Wynik częściowy lub odmowa kolektora również trafia do archiwum. Błąd kontroli źródeł, interpretera lub dostępności wejść przerywa przebieg przed zebraniem danych.

Pole `measurement_status` opisuje kompletność zebranego materiału. Po zbiorczej diagnostyce kolektor odrębnie uruchamia `propose(..., v5d7=True)`. W `proposal_attempt` zapisuje dokładną odmowę albo profil, stan propozycji, jej SHA-256 i zgodność archiwów z V5d6. `v5d7-proposal.json` zawiera dokładne kanoniczne bajty propozycji. Jej status `PROPOSAL_REQUIRES_INDEPENDENT_APPROVAL` pozostawia niezależne aprobaty osobnym krokiem.

Raport obejmuje pliki METADATA i kandydatów pominięcia w postaci źródłowych bajtów, zakres i przyczyny odmów oraz źródła trzech współdzielonych plików Python. Rozróżnia `MEASURED`, `PARTIAL`, `NOT_TESTED` i `REFUSED`. Analiza ELF obejmuje klasę, maszynę, typ, role, SOABI z nazwy, `DT_NEEDED`, `RPATH`/`RUNPATH`, wersje symboli i ABI, eksporty `PyInit` i relokacje. Pomiar fizycznego hosta zbiera dostępne statyczne dane dla interpretera, loadera, bibliotek systemowych i lokalnych wiązań libvirt. Odczyt `ldconfig -p` dostarcza kandydatów z cache. Faktyczne rozwiązywanie przez loader, zgodność bibliotek w procesie i kwalifikacja ABI pozostają `NOT_TESTED`.

Przebieg jest statyczny i zachowuje oryginalne pliki. Instalacja, ładowanie native i operacje libvirt wymagają odrębnej autoryzacji. `STAGE_9=IN_PROGRESS`, `OBSERVER_3A=BLOCKED`, `STAGE_3B=NOT_AUTHORIZED`, `PRODUCTION=NO-GO`.
