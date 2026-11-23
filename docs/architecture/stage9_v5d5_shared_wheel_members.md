# Stage 9 V5d5 — dokładny kontrakt współdzielonych członków wheel

Baza: `c6dbd5acaa87edd5c6b823bfb013764e5c5be6a8`, repozytorium
`Dudzian/Dudzian`. V5d5 opisuje osobno zatwierdzaną możliwość materializacji
62 konkretnych plików pary PySide6 6.10.3. Domyślna odmowa kolizji pozostaje
obowiązkowa. Identyczne bajty, propozycja buildera i jej lokalny SHA-256 nie
stanowią niezależnej aprobaty.

## Kontrakt i niezależne oczekiwania

Rewizja polityki to `stage9-v5d5-pyside6-6.10.3-shared-members-v1`.
Moduł `scripts/stage9_v5d5_shared.py` przypina dokładne tożsamości obu
dystrybucji, oryginalnych wheels oraz wszystkich 62 członków. Nie dopuszcza
ogólnego współdzielenia identycznych plików innych pakietów ani innych wersji.

Każdy wpis wiąże wynikową ścieżkę, SHA-256, rozmiar, oryginalny tryb i typ
`regular` oraz pełny uporządkowany zestaw właścicieli. Właściciel zawiera
`distribution`, `version`, `wheel_filename`, `wheel_sha256`, `member`.
Zestaw wszystkich właścicieli jest zbierany przed decyzją o kolizjach.
Oryginalna nazwa członka w każdym wheel i ścieżka wynikowa pozostają
oddzielnymi polami: dla tej pary są identyczne. Zmieniona relokacja `.data`
nie może skorzystać z wyjątku przypisanego do innego członka archiwum.

Pełny `RECORD` obu oryginalnych archiwów jest sprawdzany przed zastosowaniem
polityki bytecode lub współdzielenia. Inspekcja kontroluje każdy SHA-256,
rozmiar, ścieżkę, metadane wheel i typ członka, bez importowania pakietów.
Nie modyfikuje wheels ani upstream `RECORD`.

Propozycja wymienia kandydatów ze statusem
`SHARED_WHEEL_MEMBERS_REQUIRES_INDEPENDENT_APPROVAL` i trzy wymagane przeglądy
źródeł. Nie tworzy dokumentu approval. Oddzielny recenzent musi dostarczyć
dokładny kontrakt oczekiwań z polami `policy`, `members`, `source_reviews`.
`members` musi obejmować cały dokładny wykaz; `source_reviews` musi zawierać
oddzielną jawną decyzję `APPROVED_FOR_SHARED_MATERIALIZATION` dla każdego
z trzech źródeł, jego SHA-256 i niepustą niezależną `review_reference`.
Niekompletność, dodatkowy członek, trzeci właściciel, zmiana któregokolwiek
wheel, bajtów, rozmiaru, trybu, typu lub mapowania oznacza odmowę.

Oczekiwania, referencję decyzji i jej hash należy uzyskać niezależnym kanałem.
Lokalne porównanie nie uwierzytelnia recenzenta. Tabela przypięć w kodzie
identyfikuje wyłącznie dopuszczalnych kandydatów; nie jest aprobatą ich
materializacji. [Niniejszy raport statyczny](evidence/stage9_v5d5_shared_wheel_inspection.json)
również ma `approval=NOT_GRANTED` i nie może zastąpić tego kontraktu.

CLI analizy używa jawnego `--allow-shared-candidates` wyłącznie podczas
przygotowania propozycji; bez niego dotychczasowe wywołanie nadal odmawia
kolizji. Weryfikacja przyjmuje odrębny plik przez `--approved-shared`.
Przełącznik propozycji nie zatwierdza współdzielenia.

Dotychczasowe oczekiwania dotyczące źródeł kontrolera, całego domknięcia
zadeklarowanych zależności, bibliotek natywnych, ABI, runtime bazowego
i dwóch pominięć bytecode pozostają obowiązkowe. Nie usunięto PySide6
ani nie ograniczono zależności do bibliotek bezpośrednio importowanych
przez Observer.

## Plan materializacji i receipt

Po dokładnym porównaniu niezależnych oczekiwań builder przygotowuje pełny
plan przed pierwszym zapisem. Dla współdzielonej ścieżki weryfikuje deklaracje
każdego właściciela względem oryginalnych bajtów archiwum, a następnie
wybiera deterministycznie jedną materializację. Zmiana kolejności wejściowych
wheels nie zmienia właścicieli ani planu. Kolizje wewnątrz pojedynczego
archiwum, kolizje plik/katalog, z kodem kontrolera lub bindingiem libvirt
pozostają zabronione.

Katalog wynikowy musi być nowy. Zapisy używają trybu wyłącznego, nie nadpisują
istniejących plików i nie dopuszczają dowiązań ani niekontrolowanego scalania.
Po materializacji sprawdzany jest pełny wykaz i wszystkie bajty katalogu,
również dodatkowe lub brakujące ścieżki. Receipt zachowuje kompletną relację
właściciel–członek archiwum–plik wynikowy, zatwierdzone oczekiwania, hashe
wynikowe i rewizję polityki. Launcher ponawia kontrole spójności.

Pole `approval.wheel_shared_members` przenosi niezależne oczekiwania do
buildera. Receipt profilu `stage9-v5d5-prepared-environment-v1` zachowuje:

- `wheel_materialization`: tożsamości wheels, oryginalny manifest członków,
  metadane każdego członka i dokładne mapowanie na ścieżki wynikowe;
- `shared_wheel_members`: pełny wykaz i wszystkie deklaracje właścicieli;
- `shared_wheel_expectations`: osobno dostarczony kontrakt oczekiwań;
- `shared_wheel_policy`: dokładną rewizję polityki V5d5;
- `shared_wheel_output`: ścieżki, SHA-256, wynikowy tryb `0o600` i typ;
- `package_files` oraz `files`: pełny manifest wynikowych bajtów.

Kopia `shared-audit.py` sprawdza kontrakt przy weryfikacji receipt i przez
launcher. Zgodność lokalnego raportu pozostaje kontrolą integralności danych,
a nie dowodem pochodzenia aprobaty.

Współdzielenie nie autoryzuje importów natywnych ani uruchomienia kontrolera.
Odrębna bramka native load i niezależne dowody V5d1 nadal obowiązują.
Lokalny receipt nie stanowi samodzielnego admission ani kwalifikacji Stage 9.
Operacje materializacji i startu tego profilu pozostają Linux-only;
pozostałe platformy wykonują przenośne kontrole statyczne i odmowy.

## Ponowna statyczna inspekcja rzeczywistych wheels

18 listopada 2026 pobrano oryginalne archiwa z dokładnych URL PyPI zapisanych
w [raporcie V5d4](evidence/stage9_v5d4_wheel_bytecode_inspection.json).
Nie instalowano ich, nie importowano pakietów, nie wykonywano źródeł,
bytecode, bibliotek natywnych ani programu z archiwum.

| Dystrybucja | Bajty wheel | SHA-256 |
| --- | ---: | --- |
| PySide6-Essentials 6.10.3 | 77077386 | `5fb32b48010f513335a2166ac3df279c5996fb96402fd91a4875fc717fe55de0` |
| PySide6-Addons 6.10.3 | 170760635 | `a9a53a824ba8429953345c89a2dd5dc65d1f1d7e88b6cf1b975a6ef4f4d9760f` |

Pełne nazwy plików to odpowiednio
`pyside6_essentials-6.10.3-cp39-abi3-manylinux_2_34_x86_64.whl` oraz
`pyside6_addons-6.10.3-cp39-abi3-manylinux_2_34_x86_64.whl`.

| Kontrola | Essentials | Addons |
| --- | ---: | ---: |
| Oryginalne pliki / dokładny wykaz `RECORD` | 2340 / 2340 | 1283 / 1283 |
| Poprawne SHA-256 i rozmiary w `RECORD` | 2339 / 2339 | 1282 / 1282 |
| Własny wiersz `RECORD`, puste hash i rozmiar | PASS | PASS |
| Pliki po dwóch przypiętych pominięciach CPython 3.11 | 2338 | 1283 |
| Natywne członki nadal wymagające osobnego review | 263 | 189 |

Każdy członek został odczytany z kontrolą CRC. Zbiory plików i rekordów są
dokładnie zgodne. SHA-256 i rozmiary obu wheels oraz wszystkie 62 wspólne
ścieżki, hashe, rozmiary, tryby i właściciele są zgodne z V5d4.
Niezmieniony weryfikator V5d4 z podanej bazy nadal odmawia wspólnej
materializacji: `WHEEL_MATERIALIZATION_COLLISION:PySide6/Qt3DAnimation.pyi`.
Nowy inspektor V5d5 odtworzył dokładnie te same 62 przypięte rekordy; odwrócenie
kolejności obu wheels zachowuje identyczny wykaz kandydatów. Sprawdzenie
współdzielenia bez niezależnej aprobaty nadal zwraca kontrolowaną odmowę.

[Pełny dowód JSON](evidence/stage9_v5d5_shared_wheel_inspection.json) zawiera
osobny wpis każdego z 62 plików oraz obu jego właścicieli, z tożsamością
oryginalnego wheel, SHA-256 członka, rozmiarem, trybem, typem i mapowaniem.
Jest dokładnie 58 `.pyi`, trzy `.py` i jeden `py.typed`. 60 członków ma
tryb `0o100644`; `PySide6/__feature__.pyi` i `PySide6/py.typed` mają
`0o100664`. Wszystkie są zwykłymi plikami. Zgodność trybów archiwum jest
osobnym warunkiem kontraktu; builder nadal nadaje własny bezpieczny tryb
wynikowy, bez uprawnień zapisu dla grupy i bez bitu wykonania tych źródeł.

Inspekcja dwóch archiwów nie weryfikuje pozostałych 67 wheels operatora.
Nie wykonano rzeczywistej instalacji, budowy docelowego środowiska,
kwalifikacji ABI/runtime ani eksperymentu VM.

## Jawny statyczny przegląd trzech źródeł

Poniższe ustalenia pochodzą z odczytu pełnych źródeł oraz parsowania ich AST.
Nie są niezależnym zatwierdzeniem do materializacji lub wykonania.

| Źródło | Bajty | SHA-256 |
| --- | ---: | --- |
| `PySide6/__init__.py` | 6220 | `fe42fff5fcd2e9291495519bd068660a2861173bd106e5f145d8b6f72c1281fe` |
| `PySide6/_config.py` | 823 | `216206d44725c34fbb9add3ad00e6df452bc3f4eec2e923a0f716607a438d071` |
| `PySide6/_git_pyside_version.py` | 693 | `bac9879d5cf3f32d412f1e94254d1a1b6b6ed2959841519fda9d5645cc561c7b` |

`__init__.py` podczas importu wywołuje `_setupQtDirectories()`, które
importuje `Shiboken` z `shiboken6`, także na Linux i macOS. Może to załadować
biblioteki natywne. Następnie wywołuje `Shiboken.replaceModuleDict`, zmieniając
słownik modułu `PySide6`. Leniwe wyznaczanie `__all__` skanuje pliki
`Qt*.so`/`Qt*.pyd` w katalogu pakietu. Gałęzie Windows dodają ścieżki DLL,
modyfikują `PATH`, a przy obecności katalogu OpenSSL mogą importować
`QtNetwork` i wywołać `QSslSocket.supportsSsl()`. Tych gałęzi nie wykonano.
W odczytanym źródle nie stwierdzono wywołań `eval`, `exec`, subprocess ani
operacji sieciowej; obowiązek oddzielnego review źródła/native pozostaje.

`_config.py` tworzy listę `built_modules` z literału przez `split` oraz
przypisuje wersje, SONAME, datę buildu i wersję pakietu. Nie zawiera importów
ani operacji plikowych, sieciowych czy native load. `_git_pyside_version.py`
importuje future annotations, przypisuje literały wersji i ma wywołanie
`print` wewnątrz strażnika `__name__ == '__main__'`. Nie uruchamiano tego
wywołania. Każdy z tych plików pozostaje źródłem Python wymagającym własnej
decyzji recenzenta w kontrakcie współdzielenia.

## Otwarte blokady

Niezależna aprobata dokładnego współdzielenia i trzech źródeł nie została
udzielona. Nadal wymagane są wszystkie dotychczasowe aprobaty źródeł,
native/ABI, dwóch pominięć bytecode, runtime i pełnego wheelhouse.
Sprawdzenie lokalnych kontraktów lub zielone CI nie zastępują tych dowodów.
Wyniki testów i stan GitHub Actions są raportowane osobno dla konkretnego
HEAD w opisie draft PR; ten dokument nie przyznaje kwalifikacji.

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```
