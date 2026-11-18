# Stage 9 — V5d4: pominięcie przypiętego bytecode obcej wersji

Uzupełnienie po pełnym CI:
[remediacja Windows/macOS/Ubuntu i wykaz wszystkich awarii](stage9_v5d4_pr3284_ci_remediation.md).
Polityka bytecode, RECORD, aprobaty i kolizji pozostaje bez zmian.

Baza: `b143b570a8159426020ddaac8c31d74059667c70`, repozytorium
`Dudzian/Dudzian`. Poprawka rozpoznaje wyłącznie dwa dokładnie przypięte pliki
bytecode CPython 3.11 z określonego wheel PySide6-Essentials 6.10.3 jako
kandydatów pominięcia przy przygotowaniu środowiska CPython 3.12.3.
Najpierw weryfikuje pełne metadane i `RECORD`, również dla tych plików.
Materializacja wymaga osobnego, jawnego zatwierdzenia wyjątku od kontraktu
instalacyjnego. Ten dokument ani kod nie nadają takiej aprobaty.

Rzeczywista para Essentials/Addons nadal odmawia wspólnej materializacji:
ma 62 kolizje, w tym trzy źródła Python. Żadnej kolizji nie dopuszczono.
Nie zmieniono wersji PySide6 ani `pyproject.toml`.

Dane operatora obejmowały 69 wheels, 416633346 bajtów i SHA-256 względem PyPI
69/69 PASS na Ubuntu 24.04.5 z CPython 3.12.3. Ta inspekcja pobrała osobno
dwa wskazane archiwa PySide6; nie odtwarza pozostałych 67 wheels ani pomiaru
fizycznego hosta.

## Statyczna inspekcja rzeczywistych archiwów

Pobrano niezmienione archiwa PyPI. Nie instalowano ani nie wykonywano ich kodu, nie importowano
pakietów, nie deserializowano marshal. [Dowody
maszynowe](evidence/stage9_v5d4_wheel_bytecode_inspection.json) zachowują metadane, dane obu plików
bytecode i wszystkie kolizje. Pełen inwentarz członków z SHA-256 zmierzono podczas inspekcji; nie
powielono go w repozytorium. Inspektor odtwarza pełny manifest `materialized_members` z oryginalnych
wheel.

| Dystrybucja | Rozmiar wheel | SHA-256 | Pliki / sprawdzone SHA-256 RECORD |
| --- | ---: | --- | ---: |
| PySide6-Essentials | 77077386 | `5fb32b48010f513335a2166ac3df279c5996fb96402fd91a4875fc717fe55de0` | 2340 / 2339 |
| PySide6-Addons | 170760635 | `a9a53a824ba8429953345c89a2dd5dc65d1f1d7e88b6cf1b975a6ef4f4d9760f` | 1283 / 1282 |

RECORD ma dokładnie pełny wykaz plików; własny wiersz ma wymagane puste hash i rozmiar. Wszystkie
pozostałe wiersze SHA-256 i rozmiary PASS, także dla obu .pyc. Każdy członek odczytano z kontrolą
CRC i zgodności nazwy lokalnego nagłówka z centralnym katalogiem ZIP. Brak duplikatów nazw,
symlinków, specjalnych plików i niebezpiecznych ścieżek. Nagłówki lokalne i centralne zgadzają się
co do flag, metody, CRC i rozmiarów. Archiwa są jednodyskowe, wszystkie rekordy lokalne są ciągłe od
offsetu 0 do katalogu centralnego, poprawny EOCD kończy plik, brak prefiksu/trailera/luk.

METADATA i nazwy wheel/dist-info zgadzają się dla wersji 6.10.3. WHEEL: 1.0, Root-Is-Purelib=false,
Tag=cp39-abi3-manylinux_2_34_x86_64. Requires-Python <3.15,>=3.9. Essentials zależy od
shiboken6==6.10.3, Addons także od PySide6_Essentials==6.10.3. Stabilny ABI dotyczy rozszerzeń
natywnych; nie oznacza zgodności bytecode 3.11 z CPython 3.12.3.

## Dwa pliki bytecode

| Ścieżka | SHA-256 | Bajty | Nagłówek 16 B (hex) |
| --- | --- | ---: | --- |
| PySide6/support/__pycache__/__init__.cpython-311.pyc | `5e8a29507803d23847f5444596f792cbf95aa49ece2217cac2cd62489459a883` | 353 | `a70d0d0a00000000dd6ccb69df000000` |
| PySide6/support/__pycache__/deprecated.cpython-311.pyc | `a5f978d1d2e82a5d08e4a1bbe3a098e486ce601ed18ac1da4ba359aac02d5a02` | 293 | `a70d0d0a00000000dd6ccb695d020000` |

Oba mają magic a70d0d0a, czyli format CPython 3.11 (3495), flags=0 i deklarowany mtime 1774939357
(2026-03-31 06:42:37 UTC). Referencja: [CPython v3.11.0,
`Lib/importlib/_bootstrap_external.py`](https://github.com/python/cpython/blob/v3.11.0/Lib/importlib/_bootstrap_external.py#L420),
blob Git `f603a89f7f5a98c5081df371ee6f76aed4df7f79`. Format magic nie pozwala ustalić dokładnego
wydania poprawkowego interpretera ani kompilatora, który wytworzył bytecode. Pochodzenie bajtów jest
ograniczone do przypiętego wheel z PyPI; nie ma dowodu odtwarzalnej kompilacji.

Oba odpowiadające źródła istnieją, są objęte pełnym RECORD i mają rozmiary zgodne z nagłówkiem:

- PySide6/support/__init__.py: 223 B, SHA-256 `544cd819bb55076c22ad952087d0dcef414dc64f6f09a4ac65ad1ee0216b5695`.
- PySide6/support/deprecated.py: 605 B, SHA-256 `95431e9fc9d51387f56f2b1a75f38be22794a4addcd7c416f09b515ac0c35506`.

Obecność, rozmiar i timestamp nie dowodzą semantycznej równoważności pyc i źródła. Bezpieczna
polityka może pominąć wyłącznie te dwa przypięte hashe w tym przypiętym wheel po niezależnej
aprobacie wyjątku, zachowując oryginał i pełne RECORD.

Nie znaleziono innych .pyc, .pyo, .pth, sitecustomize.py ani usercustomize.py w obu wheel. Nie
oznacza to aprobaty pozostałego kodu: Essentials zawiera 263 ELF i 42 źródła .py, Addons 189 ELF i 7
źródeł .py. Wszystkie pozostają w dotychczasowym przeglądzie kodu/native ABI. W inwentarzu
Essentials 303 pliki mają ustawiony bit wykonania, Addons 174; obejmują ELF, źródła i pięć zasobów
danych. Builder nadal nadaje własne tryby i nie powinien traktować bitu x jako aprobaty.

## Kolizje Essentials / Addons

Jest dokładnie 62 wspólnych ścieżek plików. Wszystkie mają identyczne bajty SHA-256 i identyczny
tryb w obu archiwach: 58 .pyi, trzy .py oraz py.typed. Pełny wykaz każdej
ścieżki/rozmiaru/mode/SHA-256 jest w [raporcie
maszynowym](evidence/stage9_v5d4_wheel_bytecode_inspection.json). Brak globalnego wyjątku dla tych
kolizji. Obejmują wykonywalne podczas importu źródła:

- PySide6/__init__.py: 6220 B, SHA-256 `fe42fff5fcd2e9291495519bd068660a2861173bd106e5f145d8b6f72c1281fe`.
- PySide6/_config.py: 823 B, SHA-256 `216206d44725c34fbb9add3ad00e6df452bc3f4eec2e923a0f716607a438d071`.
- PySide6/_git_pyside_version.py: 693 B, SHA-256 `bac9879d5cf3f32d412f1e94254d1a1b6b6ed2959841519fda9d5645cc561c7b`.

Identyczne bajty nie stanowią aprobaty ani autoryzacji współdzielenia ścieżki. Domyślna odmowa
kolizji powinna pozostać; każde przyszłe współdzielenie wymaga oddzielnej analizy właścicieli,
mapowania/materializacji i niezależnego zatwierdzenia. Naprawa bytecode ujawni tę kolejną blokadę
rzeczywistego wheelhouse.

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```

## Osobny wykaz 62 niezatwierdzonych kolizji

Każdy wiersz oznacza osobną kolizję dwóch dystrybucji: `pyside6-essentials` i `pyside6-addons`, obie
w przypiętej wersji 6.10.3. Hash i rozmiar są identyczne po obu stronach; efekt we wszystkich
przypadkach: `REFUSED_REQUIRES_SEPARATE_ANALYSIS`. Nie przyznano wyjątku. `.py` może być wykonywane
przy imporcie; `.pyi` i `py.typed` są zasobami typowania, ale nadal podlegają kontraktowi kolizji i
właścicielstwa.

| Ścieżka | Bajty | SHA-256 w obu wheel | Kategoria |
| --- | ---: | --- | --- |
| PySide6/Qt3DAnimation.pyi | 30930 | `ee3b1c65b2ff4ccfb69db77a2b3e287cffee9ae4b19a483b31914289ac8a8625` | dane typowania |
| PySide6/Qt3DCore.pyi | 35578 | `abeefd82a06c54e0ceb811e7a6432868fc6f31a930ae0d101b359aa050b5b87f` | dane typowania |
| PySide6/Qt3DExtras.pyi | 75668 | `df36057ecb02dc736dc69a949bcbd1bcd40bf2a0ceb46b3b26939cb9421b5706` | dane typowania |
| PySide6/Qt3DInput.pyi | 23767 | `2efae1da887b5a8414c1365a60a93759cfb5511ee454a6150c790451956779e4` | dane typowania |
| PySide6/Qt3DLogic.pyi | 994 | `6e12a6c870387d6ed37785dd36347c3a799040a6b28af37ea644f3fde7d22fbf` | dane typowania |
| PySide6/Qt3DRender.pyi | 147702 | `fcf5352dc5d4d1463a589f61493757852e33981e5ac1bf3142f7fa926e08816a` | dane typowania |
| PySide6/QtBluetooth.pyi | 72822 | `900ff951a4a9df65c654dd5fabf3b0f488ceeaec0e65855c2d6ef0422036a4e8` | dane typowania |
| PySide6/QtCharts.pyi | 106955 | `a058788c9d2782b02477411c6c42b6fef501a9137ba4ace87048c22cc8835f41` | dane typowania |
| PySide6/QtConcurrent.pyi | 3817 | `a3f978e025c2f8f850d8c1b979fd47781d509729e58862eda92038ba5b22a3d5` | dane typowania |
| PySide6/QtCore.pyi | 552930 | `0a8c9dfda27cede5051cabfd582570bf26e1051d5d30bee6114b359627391803` | dane typowania |
| PySide6/QtDBus.pyi | 33933 | `3172df3ac2b505ebf040ca6160cf4b0a9b650bb3755d5476f6abca06c00e765a` | dane typowania |
| PySide6/QtDataVisualization.pyi | 114681 | `f3e4259937840185eeb8a8b53bfe01bec2845f79fc5d8aaa34995d023d20d233` | dane typowania |
| PySide6/QtDesigner.pyi | 29943 | `5f32fc2fc4d4290144936f60e825be6d8820530f8afab883410d78abeb6c5f4e` | dane typowania |
| PySide6/QtGraphs.pyi | 154716 | `d820798c931b8783cf8a026e43c1f9c9334fb9cff21d03c771bfd0d1296bf82d` | dane typowania |
| PySide6/QtGraphsWidgets.pyi | 20508 | `02d9fe330ac5aa3749a1a197cc5969ec4bdcb932aba6154e692c10c65adefe29` | dane typowania |
| PySide6/QtGui.pyi | 625997 | `2a597735578534cb7275510e072731e7728f13ea7f4935ce824eb4a61750a718` | dane typowania |
| PySide6/QtHelp.pyi | 16618 | `f0cce35edff64b0373ab510a47f8b7a288274fbc2e4aa061082c3020134fac46` | dane typowania |
| PySide6/QtHttpServer.pyi | 15355 | `17669881470c1da13a83d896d635ea09377d2799695347e9985338ead707db52` | dane typowania |
| PySide6/QtLocation.pyi | 57311 | `cb059fe355c91bf697f489eec4bbe310e5d499af13e518f91283c160ef4a0428` | dane typowania |
| PySide6/QtMultimedia.pyi | 88555 | `c81b630413ca09a7e2c60cb379ae0667bf37d2cf5c64e856757088b0c0094d5c` | dane typowania |
| PySide6/QtMultimediaWidgets.pyi | 3339 | `1af5ec4a8fbe91fc67c64019818e07fbefed506212da23f1864bd2399c7c3d10` | dane typowania |
| PySide6/QtNetwork.pyi | 147030 | `ab4cca9fd5239ecb811392d90934908b34ee540c2fc4587f3d7ca1d1f8a43769` | dane typowania |
| PySide6/QtNetworkAuth.pyi | 27670 | `6f6611cd03b7bf2b2831432a878bd69edd49bcc5596861a95936b48616593b76` | dane typowania |
| PySide6/QtNfc.pyi | 18368 | `8bf5b10d2cfc6832e1eb2ab84084e474ac6a49a1136cc86d3d233dfad1202736` | dane typowania |
| PySide6/QtOpenGL.pyi | 1206589 | `6720bf58471ac3820fa1b166a7c5368352330b2da3bf9fca918a1d8c8e9a7a5f` | dane typowania |
| PySide6/QtOpenGLWidgets.pyi | 3264 | `518400b2516f2bca39a554acae2acdbb17f307e38a5d4b0fef2db69b300271c5` | dane typowania |
| PySide6/QtPdf.pyi | 16191 | `9574a99062e491da8fb7eb14ec3f7f22776b4a5b2fade39fa4eaf73f757d10fa` | dane typowania |
| PySide6/QtPdfWidgets.pyi | 5281 | `15ec6774a171f740f00a8c3e9ea5ec6e55dc0a8818e56821444d2a8002b93fee` | dane typowania |
| PySide6/QtPositioning.pyi | 38444 | `b29a7375e8dd366a8437bae4d97f73d4d4d3b2d06e894b31bd0648a35b76356f` | dane typowania |
| PySide6/QtPrintSupport.pyi | 16889 | `9d953419e558fdf930139c6715fe70c9f384dab2a9104d0ad58ce51092eec5ce` | dane typowania |
| PySide6/QtQml.pyi | 51704 | `d0bef60b580a75515edeaa9c4c2cdfeb76340bb9da5b455dad071e79f869e20e` | dane typowania |
| PySide6/QtQuick.pyi | 88751 | `42c4a63e80323e3a9f761bab3a3b6767193be361a15e19317d9ecd5069ad78f2` | dane typowania |
| PySide6/QtQuick3D.pyi | 14348 | `08d62baeb5a7c3f9de7f77f05c44eaf00ca0d1aa2caccaad20a4aae5eaf0cdb9` | dane typowania |
| PySide6/QtQuickControls2.pyi | 1370 | `3e07d01d0144d883f78d867f3b93c3383a4363f2f980b281e1285d87bbead28f` | dane typowania |
| PySide6/QtQuickTest.pyi | 694 | `b0977987bbfdea5953e871f7b2ddff971d50dea40ef197de28eac18f39e94a01` | dane typowania |
| PySide6/QtQuickWidgets.pyi | 5471 | `d3caa79cd9efd5a9440b737e358e2f423199025337a52a14e3b9f20f4ae7f930` | dane typowania |
| PySide6/QtRemoteObjects.pyi | 19499 | `7c31cb6bae4f2d7276a9aac7d6008f2fc4798c2c074b66c7bf73f622aa78819e` | dane typowania |
| PySide6/QtScxml.pyi | 17522 | `e7a769fa9637759e3dc0b311ed9dedc936320089399686e86ceb5c6cf9da7e5c` | dane typowania |
| PySide6/QtSensors.pyi | 27936 | `6233707ce0df307ef6facba50b54378552bd69a98b59e866d9e7f27e257e7fbf` | dane typowania |
| PySide6/QtSerialBus.pyi | 37587 | `27597e0c768cbed3c98cdcef83b06c1197f7b9c514c954d2fd990664a8548e2f` | dane typowania |
| PySide6/QtSerialPort.pyi | 9541 | `105ae6fe2c88c1f678b7e660b3b1063dd76058897568c81656a159073f6dbbfa` | dane typowania |
| PySide6/QtSpatialAudio.pyi | 12209 | `8f1bcfdd5513c93520ebc2f455b89d22c5683fa80b64508c5e9c8db416a5dd6b` | dane typowania |
| PySide6/QtSql.pyi | 34228 | `44ed3a1e86b8e131516261b03f9fadca708de3c7b299188fa8bd400dda81f77b` | dane typowania |
| PySide6/QtStateMachine.pyi | 17056 | `a136d181e090e6d409c3609e6714671a63cb99ec57a8099a7a65950a1265d4af` | dane typowania |
| PySide6/QtSvg.pyi | 6746 | `1d2f7b7f0e49286232a2d269638d0834def5131afbf0b0d7189824c5f111489a` | dane typowania |
| PySide6/QtSvgWidgets.pyi | 2574 | `04d9ae9008f7b3248e5a7b8ececff8573ca2cd9b99049e41bcac91fe937cc45c` | dane typowania |
| PySide6/QtTest.pyi | 21983 | `b6d253062f958528635e0312f5cf381a18e3cae9f05f1dccd608d64a33c52da2` | dane typowania |
| PySide6/QtTextToSpeech.pyi | 9852 | `dced47228b32ec1bad29320d4a1f58229ff63852b0cbdda411a05088db546623` | dane typowania |
| PySide6/QtUiTools.pyi | 2347 | `86f1dbf95fbb6d64f8cbdc319eb83a633bea08e64e2d7a10f99c01f323a41619` | dane typowania |
| PySide6/QtWebChannel.pyi | 1948 | `ca69e4cf3879fd211e6cb303f648fdba7f610fddad6d399cba3128ce59a57752` | dane typowania |
| PySide6/QtWebEngineCore.pyi | 72637 | `bcef577be7b4d63fb9a7c1336c54c5ccbe58c501411a6d2cf2d1f9f30d5a1205` | dane typowania |
| PySide6/QtWebEngineQuick.pyi | 9035 | `147d416bc53c50e252ba2993713579d310317c3b17229fdc41de1218e6ff638a` | dane typowania |
| PySide6/QtWebEngineWidgets.pyi | 6359 | `f77a57e20d1e40c093cfaa4d6284533b746d62e96424c6a5ad18c58054db2dbf` | dane typowania |
| PySide6/QtWebSockets.pyi | 12630 | `d5016bf161daef5272b43d767e16ae3146c1d03fe8ffbab460c881dc37fb3bd5` | dane typowania |
| PySide6/QtWebView.pyi | 548 | `1da437dc95e285037fe9c02e785f6cf58847f07f3076b08a65f34c7c7b0f3dfd` | dane typowania |
| PySide6/QtWidgets.pyi | 603070 | `88691b6e26b78707e39d2279911da76b9be2aeb76573abb0f85122bba950ca39` | dane typowania |
| PySide6/QtXml.pyi | 20166 | `9eb600340820b022d2959d03a723d1423655ba12b624f6578339cf79f0061c5c` | dane typowania |
| PySide6/__feature__.pyi | 246 | `3f3f79ca62c186e1c5d07b228a6db57809c184804449426bda4cd31f269072e5` | dane typowania |
| PySide6/__init__.py | 6220 | `fe42fff5fcd2e9291495519bd068660a2861173bd106e5f145d8b6f72c1281fe` | źródło Python / import |
| PySide6/_config.py | 823 | `216206d44725c34fbb9add3ad00e6df452bc3f4eec2e923a0f716607a438d071` | źródło Python / import |
| PySide6/_git_pyside_version.py | 693 | `bac9879d5cf3f32d412f1e94254d1a1b6b6ed2959841519fda9d5645cc561c7b` | źródło Python / import |
| PySide6/py.typed | 33 | `154d2b3a1255946ec4511a7952fdb07cfbe01a6ee9c7c6eae5dfabd460c833c1` | dane typowania |

## Jawna polityka i wyjątek kontraktu instalacyjnego

Polityka `pyside6-essentials-6.10.3-cp311-cache-omission-v1` jest kandydatem
wąskiego wyjątku. Wymaga równocześnie dokładnej nazwy i wersji dystrybucji,
nazwy i SHA-256 oryginalnego wheel, dokładnego zbioru dwóch ścieżek `.pyc`,
SHA-256 obu plików, pełnego 16-bajtowego nagłówka, magic `a70d0d0a`, flags=0,
obecności i dokładnego SHA-256 obu źródeł oraz zgodności rozmiaru źródła
z nagłówkiem. Profil docelowy pozostaje CPython 3.12.3.

Nie ma reguły dopuszczającej dowolne `__pycache__`, `.pyc` ani wszystkie pliki
o tagu `cpython-311`. Dodatkowy `.pyc`, bytecode CPython 3.12, brak źródła
lub jakakolwiek zmiana przypiętych bajtów oznaczają odmowę. `.pth`, `.pyo`,
startup customization i inne pliki w `__pycache__` pozostają odrzucane.
Obce formaty wykonywalne PE/Mach-O i nieobsługiwane skrypty także odmawiają.
Zwykłe źródła `.py` pozostają w pełnym manifeście i wymagają dotychczasowego
przeglądu; rozpoznanie formatu nie jest aprobatą ich kodu.

Oryginalny wheel pozostaje niezmieniony. Jego `RECORD` obejmuje oba pomijane
pliki i jest w całości weryfikowany przed zastosowaniem polityki startup.
Skopiowany `RECORD` zachowuje bajty upstream i opisuje archiwum, a nie wykaz
plików po materializacji. To jawna różnica względem instalacji wszystkich
członków wheel, wymagająca niezależnej zgody `OMIT_PINNED_FOREIGN_BYTECODE`.
Nie wolno traktować tego `RECORD` jako dowodu obecności pominiętych plików
w docelowym katalogu ani go przepisywać, aby ukryć różnicę.

Dokładny wykaz docelowych plików stanowią `package_files` i `files` w raporcie
`ENVIRONMENT.json`. Pola `wheel_archives`, `wheel_omissions`,
`installation_contract_exception` i
`wheel_record_semantics=UNCHANGED_UPSTREAM_ARCHIVE_RECORD_NOT_INSTALLED_INVENTORY`
wiążą je z niezmienionym archiwum i jawnym pominięciem.

## Spójna ścieżka od inspekcji do raportu

| Etap | Kontrola i wynik |
| --- | --- |
| 1. Inspekcja i metadane | `stage9_v5d4_dependencies.py` sprawdza pochodzenie/SHA-256 wheel, strukturę i nazwy ZIP, typy plików, limity, `Metadata-Version`, tożsamość i tagi wheel, `Requires-Python`, pełny zbiór `RECORD`, SHA-256 i rozmiary. Reguła startup/bytecode jest stosowana po pełnym `RECORD`; strukturalnie niebezpieczne lub nieczytelne wejście jest odmową. |
| 2. Propozycja | `propose()` odtwarza zależności i pełną mapę `materialized_members`: członek archiwum → ścieżka docelowa i SHA-256. Osobne `omitted_members` zawiera dokładne dwa pominięcia. Pole `installation_contract_exception` ma wartość `OMIT_PINNED_FOREIGN_BYTECODE_REQUIRES_INDEPENDENT_APPROVAL`; `authorization=NOT_AUTHORIZED`. Kolizje docelowe odmawiają już tutaj. |
| 3. Niezależne oczekiwania | `verify()` odbudowuje całą propozycję z oryginalnych wejść, porównuje niezależny hash kanonicznej propozycji, pełny zatwierdzony wykaz native oraz osobny pełny obiekt oczekiwań pominięć. Brak, nadmiar albo zmiana wpisu daje `UNAPPROVED_WHEEL_OMISSIONS`. Zgodność pozostaje lokalną kontrolą; narzędzie nie uwierzytelnia recenzenta. |
| 4. Materializacja | `stage9_v5d4_environment.py` ponownie wywołuje ten sam inspektor na bajtach wheel i porównuje cały pakiet z propozycją. Wspólny plan obejmuje wszystkie wheels, źródła kontrolera i bindingi; kolizje plik/plik i plik/katalog odmawiają przed pierwszym zapisem. Pomijane pliki nie są kopiowane. Każdy zapis jest nowym plikiem; nie ma nadpisywania ani scalania identycznych bajtów. |
| 5. Katalog docelowy i ABI | `verify_materialized()` porównuje cały katalog pakietów z planem i odrzuca dodatkowe/brakujące pliki, symlinki, startup hooks i bytecode. `stage9_v5d4_abi.py` nadal odrzuca bytecode i cache w import roots, sprawdza rzeczywiste kopie plików, pełen jawny wykaz native oraz dotychczasowe oczekiwania ABI. Pominiętego pliku nie można dopuścić ponownie w docelowym drzewie. |
| 6. Raport integralności | Dopiero po zgodności plików i `STATIC_ABI_MATCH` powstaje `ENVIRONMENT.json`, z pełnym wykazem wynikowych plików, archiwów, pominięć i zatwierdzonego wyjątku. Częściowy wynik nie ma raportu sukcesu. `authorization=NOT_AUTHORIZED` i `native_load=NOT_EXECUTED` pozostają bez zmian. |

`.data/purelib` i `.data/platlib` używają tego samego mapowania w inspektorze
i builderze, z kontrolą właściciela `.data` i kolizji po mapowaniu.
Nieobsługiwane schematy `.data` odmawiają; relokacja native z `.data` nadal
wymaga osobnego projektu i jest odrzucana. ABI nie otrzymuje zwolnienia
z kontroli `.pyc`.

Ponowna inspekcja rzeczywistych wheel przez zmieniony inspektor dała:
Essentials — 2338 członków do materializacji, 263 native i dwa pominięcia;
Addons — 1283 członków do materializacji, 189 native i zero pominięć.
Wspólny plan odmawia
`WHEEL_MATERIALIZATION_COLLISION:PySide6/Qt3DAnimation.pyi`.
Nie wykonano materializacji tych rzeczywistych pakietów ani budowy
rzeczywistego środowiska docelowego.

## Procedura niezależnego przeglądu pominięć

Recenzent musi otrzymać oryginalne archiwa i niezależnie ustalić ich
pochodzenie, hash, metadane, pełny `RECORD`, obie ścieżki bytecode i źródła.
Sprawdza ograniczenie do obcego formatu CPython 3.11 oraz powyższą zmianę
semantyki instalacji. Osobno przegląda pełny kod/native i wszystkie kolizje.
Samo SHA-256 propozycji ani samo zatwierdzenie native nie upoważniają
do pominięcia plików.

CLI weryfikacji ma dodatkowy argument `--approved-omissions`, wskazujący
plik oczekiwań dostarczony przez recenzenta. API `verify()` przyjmuje go jako
`approved_omissions`. Zaufany builder przyjmuje opcjonalne pole
`approval.wheel_omissions`; gdy propozycja zawiera pominięcia, obecność
i dokładna zgodność tego pola są obowiązkowe. Pozostałe wymagania aprobaty
i zewnętrzny hash całego dokumentu approval pozostają w mocy.

Obiekt oczekiwań ma dokładnie pola:

- `policy`: `pyside6-essentials-6.10.3-cp311-cache-omission-v1`.
- `installation_contract_exception`: `OMIT_PINNED_FOREIGN_BYTECODE`.
- `members`: dokładny uporządkowany wykaz dwóch obiektów. Każdy zawiera
  `distribution`, `version`, `wheel_filename`, `wheel_sha256`, `member`,
  `sha256`, `source`, `source_sha256`, `cache_tag`, `magic`, `policy`.

Każda ścieżka i SHA-256 z tego wykazu są przypięte do ustaleń recenzenta;
wartości w propozycji są wyłącznie materiałem do porównania. Oczekiwania,
referencja zatwierdzenia i ich hashe muszą przyjść niezależnym kanałem.
Nie generować dokumentu approval z propozycji jako substytutu decyzji,
nie uznawać lokalnie wyliczonego hasha za niezależną aprobatę i nie nadawać
jej przez uruchomienie helpera `omission_expectation()`. Helper wyraża
strukturę porównania, nie decyzję recenzenta.

Nawet poprawna zgodność oczekiwań daje
`CALLER_SUPPLIED_EXPECTATIONS_NOT_AUTHENTICATED` i `NOT_AUTHORIZED`.
Tożsamość/autorytet recenzenta, chronione przekazanie oczekiwań oraz
zatwierdzenie kontraktu pozostają poza mechanizmem porównania. W tym PR
nie utworzono niezależnej aprobaty ani zwolnienia dla kolizji.

## Testy i ograniczenia wykonania

Testy polityki używają syntetycznych wheel z pełnym `RECORD`, zaufanych
narzędzi i katalogów tymczasowych. Próby pozytywne modelują dokładny
zbiór dwóch plików CPython 3.11; testowe piny fixture są oddzielone
od niezmienionych produkcyjnych pinów rzeczywistego wheel. Próby odmowy
obejmują nieprawidłowy hash lub rozmiar `RECORD`, brak źródła, dodatkowy
bytecode, bytecode CPython 3.12, `.pth`, startup customization, nieprawidłowe
ścieżki, symlinki, specjalne pliki, niespójny plan, kolizje oraz niekompletny
wykaz niezależnie oczekiwanych pominięć. Sprawdzana jest ścieżka od inspektora
przez weryfikację oczekiwań do rzeczywistego katalogu tymczasowego.

Końcowy przegląd uszczelnił także granicę oczekiwań: dokumenty approval,
ABI request i ABI expectations są parsowane z tych samych bajtów, których
hash wcześniej sprawdzono. Usunięto ponowny odczyt mogący podmienić
zatwierdzoną treść. Po analizie ABI końcowy wykaz `packages/` musi nadal
odpowiadać zatwierdzonemu planowi; builder nie może uznać nowych bajtów
za oczekiwane przez samo zapisanie ich w receipt. Regresje podmiany danych,
usunięcia plików i dodatkowego pliku wymagają odmowy bez `ENVIRONMENT.json`.
Kontrole snapshotu nie ustanawiają immutable view ani nie usuwają wyścigu
po zakończeniu pomiaru.

## Walidacja lokalna — 2026-11-17

Baza: `b143b570a8159426020ddaac8c31d74059667c70`. Narzędziowy CPython
3.12.14, Ruff 0.14.11, mypy 1.10.0, Semgrep 1.179.0 i Betterleaks 1.9.0
działały w odrębnym katalogu roboczym poza repo. Nie instalowano badanych
pakietów; narzędziowy interpreter nie jest fizycznym CPython 3.12.3 operatora.

| Kontrola | Wynik |
| --- | --- |
| V5d4 dependencies | 132 PASS |
| V5d4 environment | 64 PASS |
| V5d4 ABI | 55 PASS |
| V5d4 closure | 102 PASS |
| V5d3 release | 255 PASS |
| V5d2 inventory i historia CI | 65 PASS po powtórzeniu |
| V5d1: collector identity, domain, image, preflight | 458 PASS, 1 SKIP |
| Ruff — pełny dotychczasowy zakres | PASS |
| Dokładny hook `python -m pre_commit run ruff-format --all-files` | PASS |
| `python -m pre_commit run --all-files` | PASS |
| Configured mypy Linux i Win32 | PASS, po 424 pliki |
| Blokujący Semgrep / testy reguł | 0 findings / 5 z 5 PASS |
| Betterleaks całego drzewa z niezmienioną bazą / staged hook | PASS / PASS |
| Pip-audit `deploy/packaging/requirements-desktop.lock` bez dodatkowych wyjątków | PASS |
| Diff coverage względem bazy, istniejący próg 70% | 98% PASS |

Łącznie: 1131 PASS i jeden istniejący SKIP dla natywnej odmowy hosta innego
niż Linux. Zestaw V5d4/V5d3 wykonano razem (601 PASS przed siedmioma
końcowymi regresjami); końcowy V5d4 powtórzono osobno (353 PASS).
Testy używały `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, odpowiedniego
`--confcutdir=tests/deployment` lub `tests/scripts`, JUnit oraz branch coverage.

Pierwszy V5d2 miał dwa timeouty subprocess po 20 s; nie potwierdzono ich
przyczyny. Cały niezmieniony zestaw powtórzono seryjnie: 65 PASS,
bez wydłużania limitów i bez wyłączeń. Pierwszy Betterleaks wskazał zastany,
ignorowany cache testowego klucza CPython 3.11. Zachowano ten pojedynczy
wygenerowany plik poza repo i powtórzono pełny scan: zero findings.
Baseline, reguły i zakres skanera pozostały bez zmian.

Pełny SHA gałęzi, link draft PR oraz wyniki GitHub Actions dla tego SHA
są w opisie PR. Wyniki lokalne nie zastępują Actions. Nie zmieniono
istniejących bramek bezpieczeństwa, progów ani zasad niezależnej akceptacji.

Nie instalowano badanych dystrybucji podczas inspekcji, nie wykonywano ich
bytecode, źródeł ani natywnych plików. Nie używano `sudo`, nie uruchamiano
VM ani kontrolera, nie instalowano do `/opt` lub `/etc`, nie zmieniano
oryginalnych wheel i nie wykonano merge. Brak natywnej kwalifikacji hosta,
runtime i pluginów pozostaje ograniczeniem.

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```
