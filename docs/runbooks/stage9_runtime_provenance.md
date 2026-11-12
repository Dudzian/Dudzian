# Stage 9: odczytowe rozliczenie zależności systemowych

Status wykonania na właściwym hoście/VM: **BLOCKED**. S3 **BLOCKED**;
PRODUCTION **NO_GO**. Ta procedura nie podpisuje, nie zatwierdza ani nie
przebudowuje runtime. Zielone testy kolektora nie ustanawiają pochodzenia
pakietów systemowych.

Punktem odniesienia jest audyt commita
`fc15b98298c5430286dd2206ec7eeb8f20e2497b`: 1352 wpisy `external_files`,
573 pliki `.pyc` oraz 19 innych nierozstrzygniętych plików. W środowisku
remediacji nie dostarczono odpowiadającego im podpisanego manifestu, listy
tych 19 ścieżek, oryginalnych `.deb`, metadanych archiwum ani dostępu do
hosta, którego dotyczy audyt. Liczby są celami rozliczenia z audytu; nie są
tutaj ponownie potwierdzonym pomiarem. Nie wolno wywnioskować autentyczności
pozostałych wpisów wyłącznie przez odejmowanie liczebności.

## Warunki i granica odczytu

Operator identyfikuje dokładny host, VM, boot ID, wersję Ubuntu/architekturę,
runtime, niezależny autorytet Ed25519 i chronione drzewo narzędzi po remediacji.
Narzędzia uruchamia systemowy, niezależnie zaufany interpreter z `-I -B`.
Nie uruchamia interpretera, skryptów, `.pyc`, ELF ani `ldd` z badanego payloadu.
Procedura dotyczy plików, do których operator ma uprawniony odczyt; nie tworzy
kont, approval, konfiguracji, kluczy, podpisów, usług ani klastra.

Wszystkie wyniki kierować na stdout/stderr i zachować po stronie kolektora
poza badanym drzewem. Odczyt może zmienić atime; wymagana jest migawka lub
montowanie tylko do odczytu, jeśli także atime musi pozostać identyczny.
Brak spójnej migawki oznacza konieczność porównania stanu przed/po i BLOCKED
przy niejednoznaczności. Nie wykonywać `apt update/install`, `dpkg --configure`,
`compileall`, `py_compile`, zmiany manifestu, chmod/chown repozytorium ani
naprawy brakujących danych na badanym hoście.

## 1. Tożsamość manifestu przed analizą

Najpierw uruchomić istniejące odczytowe `freshness_runtime.py verify` z
chronionego drzewa. Zachować dokładne argv, stdout, stderr, czas UTC i exit.
Ta funkcja wykonuje `verify_ed25519` na surowych bajtach **przed** `_json`.
Nie używać `publish`. Wymagane są autentyczne istniejące materiały podpisu;
nie tworzyć zastępczych podpisów ani kluczy na potrzeby tej procedury.

```sh
/usr/bin/python3.12 -I -B "$TRUSTED_TREE/deployment/freshness_runtime.py" verify \
  --runtime "$RUNTIME" --manifest "$MANIFEST" --signature "$SIGNATURE" \
  --trusted-key "$TRUSTED_KEY" --environment-id "$ENVIRONMENT_ID" \
  --revision "$EXACT_REVISION"
```

Zmienne oznaczają istniejące, sprawdzone absolutne ścieżki i dokładną
tożsamość ocenianego wydania. `EXACT_REVISION` to rewizja **podpisanego
runtime**, nie automatycznie HEAD gałęzi remediacji. Exit inny niż 0 blokuje
dalsze ustanawianie zaufania. Z odpowiedzi exit 0 przenieść
`manifest_sha256` do `VERIFIED_MANIFEST_SHA256`, bez edycji bajtów manifestu.
Digest jest wiązaniem wyniku niezależnej weryfikacji; sam digest wpisany przez
użytkownika nie uwierzytelnia podpisu.

## 2. Jeden wiersz dowodu dla każdego z 1352 wpisów

Uruchomić odczytowy kolektor na tym samym hoście/migawce:

```sh
/usr/bin/python3.12 -I -B "$TRUSTED_TREE/scripts/stage9_runtime_provenance.py" \
  --manifest "$MANIFEST" --verified-manifest-sha256 "$VERIFIED_MANIFEST_SHA256"
```

Kolektor porównuje digest **przed parsowaniem JSON**, odmawia duplikatów
kluczy i niekanonicznych ścieżek. Odczytuje każdy obiekt przez `lstat`, dla
plików porównuje hash, rozmiar i tryb, dla linków literalny cel, dla katalogów
typ/tryb; raportuje UID/GID i wykrywalne podmiany podczas odczytu. Nie ładuje
kodów ELF ani `.pyc`. Raport obejmuje surowe odczytowe wyniki `dpkg-query
--search` dla wszystkich ścieżek i kandydatów źródeł `.pyc`, a także dokładne
wersje, architektury i statusy pakietów z `dpkg-query --show`.

Kody wyjścia: **1 FAIL** dla błędów odczytu/integralności/schema lub liczby
innej niż 1352/573; **2 BLOCKED** dla zgodnej obserwacji wymagającej jeszcze
uwierzytelnionych źródeł. Kolektor **nigdy nie zwraca PASS**. Wynik `dpkg-query`
jest obserwacją lokalnej bazy, a nie dowodem autentycznego pochodzenia `.deb`.
Ograniczenie nazw do kanonicznych ścieżek ASCII bez globów jest jawne;
odmowy nie wolno omijać przez pominięcie wpisu.

Utworzyć poza hostem tabelę obejmującą dokładnie zbiór kluczy `external_files`:

| Pole wiersza | Wymagany dowód |
| --- | --- |
| Ścieżka i typ | Dokładna nazwa manifestu, hash/rozmiar/tryb lub cel linku |
| Stan hosta | Zgodna obserwacja, UID/GID, stabilność w czasie pomiaru |
| Pakiet | Nazwa binarna z kwalifikatorem architektury, pełne `Version`, `Architecture`, status z dpkg |
| Archiwum | Suite/pocket, komponent, architektura, URL, czas pobrania metadanych, hash InRelease |
| Łańcuch | Fingerprint autorytetu, wynik gpgv, hash i rozmiar indeksu Packages |
| `.deb` | Dokładne `Filename`, `Version`, `Architecture`, `Size`, `SHA256` ze zweryfikowanego Packages |
| Payload | Wpis data.tar: typ, tryb, cel linku lub hash bajtów; porównanie z manifestem/hostem |
| Odstępstwo | Jawny dowód transformacji albo BLOCKED; nigdy samo `dpkg -S` |

Sprawdzić jednoznaczność ścieżek i rozliczenie zbioru w obie strony: brakujące,
dodatkowe i powtórzone wiersze to FAIL. Osobno zachować **dokładne nazwy 19
plików z audytu** i przypisać każdemu ostateczny dowód lub BLOCKED. Brak
oryginalnej listy 19 nie pozwala zastąpić jej dowolną nową grupą 19 obiektów.
Nie zakładać, że katalog, symlink, plik generowany lub konfiguracja ma takie
samo pochodzenie jak sąsiedni `.py` należący do pakietu.

## 3. Uwierzytelnienie istniejących `.deb`

To etap analizy **poza badanym hostem**, na kopiach zachowanych materiałów.
Nie wykonuje instalacji ani skryptów maintainer. Wymagana jest niezależnie
zaakceptowana kotwica Ubuntu Archive, w tym fingerprint i pochodzenie samego
keyringu. Kopia keyringu z badanego hosta nie jest samodzielną kotwicą.

1. Zachować `/var/lib/dpkg/status`, odpowiednie pliki `.list`/`.md5sums`,
   dywersje, źródła APT i preferences, istniejące `/var/lib/apt/lists` oraz
   cache `.deb` jako obserwacje. Pełna wersja zawiera epoch i Ubuntu revision;
   nazwa typu `libc6` albo `openssl 3` jest niewystarczająca. Rozliczyć multiarch,
   holds i lokalne dywersje. `md5sums` nie zastępuje łańcucha SHA-256.
2. Sprawdzić istniejący `InRelease` przez niezależny `gpgv` i keyring:
   `gpgv --keyring "$TRUSTED_ARCHIVE_KEYRING" --status-fd 1 --output "$ANALYSIS/Release" "$INRELEASE"`.
   Zachować exit, `VALIDSIG`, fingerprint, surowe metadane i wyodrębnione
   podpisane Release. `$ANALYSIS` leży poza badanym runtime/hostem.
   Zweryfikować Origin/Label/Suite/Codename/Architectures/Components oraz
   podpisany czas i politykę ważności. Historyczny, wygasły snapshot wymaga
   jawnej decyzji o użyciu historycznej daty; nie wyłączać kontroli ważności
   w celu uzyskania PASS.
3. Z podpisanej sekcji SHA256 Release wybrać **dokładny** indeks Packages i
   potwierdzić jego długość oraz SHA-256. Dekompresja lokalnego `.lz4` musi
   dać bajty odpowiadające wskazanemu wpisowi `Packages`; nie uwierzytelniać
   indeksu przez jego nazwę lub lokalny stan APT.
4. Dla każdej dokładnej pary pakiet/wersja/architektura znaleźć jednoznaczny
   stanza Packages. Zweryfikować zachowany `.deb` według `Size` i SHA256.
   HTTPS i `apt-cache policy` same nie wystarczają. Brak wersji lub oryginału
   to BLOCKED. Ewentualne pobranie brakującego historycznego artefaktu jest
   osobnym etapem pozyskania dowodów do katalogu analitycznego, bez aktualizacji
   bazy pakietów i bez instalacji; odnotować URL i zachować łańcuch podpisów.
5. Odczytać ar/control.tar/data.tar jako dane, bez `dpkg -i`, chroot i skryptów.
   Chronić analizę przed traversal, linkami, duplikatami, specjalnymi obiektami
   i bombami dekompresyjnymi. Porównać nazwę/wersję/architekturę control z
   Packages. Dla każdego obiektu porównać uwierzytelniony payload z manifestem
   i hostem. Nie wystarcza wynik `dpkg --verify`.

## 4. Wszystkie 573 `.pyc` i 19 pozostałych plików

`.pyc` to wykonywalny kod, nawet gdy `dpkg -S` nie zna jego nazwy. Kolektor
wyświetla 16 bajtów nagłówka i kandydat odpowiadającego `.py`; nie używa
`marshal`, importów, wykonania bytecode ani automatycznej kompilacji.

Dla **każdego** `.pyc` zachować hash, pełny nagłówek (magic/flags i hash źródła
lub timestamp/size), optimize level i dokładne źródło. Oddzielnie wykazać
autentyczność źródła `.py`, dokładnego pakietu CPython, generatora
`py3compile`/skryptu maintainer i parametrów kompilacji, w tym `co_filename`,
invalidation mode i właściwej ścieżki. Sam magic, poprawny timestamp lub hash
źródła w nagłówku **nie uwierzytelnia** pozostałych bajtów bytecode.

Jeśli `.pyc` nie występuje w uwierzytelnionym data.tar, potrzebny jest
zachowany wiarygodny dowód jego generacji i porównanie dokładnych bajtów albo
osobno zaakceptowana analiza odtwarzalności w izolowanym katalogu analitycznym.
Ta procedura pozostaje odczytowa: nie wykonuje takiej kompilacji i nie modyfikuje
runtime. Brak dowodu daje BLOCKED dla konkretnej ścieżki; nie usuwać `.pyc`
ani nie aktualizować manifestu, żeby uzyskać zgodność.

Każdy z pozostałych 19 plików wymaga odrębnej klasyfikacji. Dla wygenerowanego
`ld.so.cache` potrzebne są uwierzytelniony `ldconfig`, wejścia konfiguracji,
pełny inwentarz bibliotek i dowód generacji; cache nie pochodzi bezpośrednio
z payloadu `.deb`. Dla conffile sprawdzić payload, dpkg conffiles, dywersje i
niezależnie autoryzowaną zmianę. Lokalnie generowane metadane lub pliki poza
dpkg wymagają własnej kotwicy, nie arbitralnego przypisania do pakietu.

## 5. Loader, libc, OpenSSL i pełne zależności natywne

Uwzględnić każdy ELF, symlink i katalog z manifestu, bazowy CPython, loader
z `PT_INTERP`, libc, libcrypto/libssl, `_ssl`, `_hashlib`, psycopg/libpq oraz
pozostałe biblioteki. Nie ograniczać się do nazw zakończonych `.so`.
Na uwierzytelnionych kopiach odczytać przez zaufany `readelf` nagłówki ELF,
`DT_NEEDED`, `RPATH/RUNPATH`, build-id i symbole wersjonowane. Rozwiązać
zależności do konkretnych uwierzytelnionych ścieżek, wersji i hashy.

Rozliczyć literalne cele symlinków oraz konfigurację loadera i cache.
`ldd` nie jest dozwolonym sposobem badania niezweryfikowanych plików.
Statyczne `DT_NEEDED` nie dowodzi domknięcia `dlopen`, NSS, lokalizacji ani
providerów OpenSSL. Potrzebne są zachowane ślady rzeczywistych operacji
`openat`/`execve`/mapowań **zatwierdzonej** próby VM, obejmującej pozytywne
i negatywne ścieżki Ed25519, start usługi i obsługę żądań. Nie uruchamiać
próby hosta/VM automatycznie w PR. Brak śladów pozostaje BLOCKED, nawet jeśli
cały odczytowy inwentarz odpowiada podpisanemu manifestowi.

Bootstrap po remediacji ma własny zamknięty kontrakt konfiguracji/providerów;
trzeba nadal rozliczyć pochodzenie użytego `/usr/bin/openssl`, loadera,
libcrypto i wszystkich faktycznych mapowań przez powyższy łańcuch `.deb`.

## Kryteria zakończenia i zachowanie dowodów

Każde polecenie zachować z pełnym argv, czasem UTC, stdout/stderr i oryginalnym
exitem. Pipeline z `tee` wymaga `pipefail` i zachowania exit polecenia, a nie
tylko `tee`. Hashe plików dowodowych służą wykrywaniu zmian; nie są podpisami
ani approval. Nie dopisywać do manifestu runtime brakujących wierszy i nie
usuwać niepasujących plików.

Lokalnie testowany jest wyłącznie kolektor: odczyt plików jako danych,
negatywne przypadki digestu/ścieżki/podmiany, brak wykonania `.pyc`, jawne
BLOCKED i zachowanie exit 1/2. Rozliczenie właściwych 1352 wpisów, pochodzenie
573 `.pyc`, dokładnych 19 plików, wersji Ubuntu i pełnego closure natywnego
pozostaje **BLOCKED do czasu uzyskania wyżej wymienionych oryginalnych dowodów**.
Przebudowa runtime wymaga wykazanej konieczności i osobnego zatwierdzenia;
nie jest skutkiem samego braku materiałów audytowych.
