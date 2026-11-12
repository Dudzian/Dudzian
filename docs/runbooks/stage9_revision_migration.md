# Stage 9 — zgodność rewizji i migracja po remediacji PR #3265

**Status wykonania migracji: BLOCKED / NOT_EXECUTED. S3: BLOCKED.
PRODUCTION: NO_GO.** Ten dokument nie jest zatwierdzeniem budowy, podpisania,
publikacji ani próby VM. W ramach poprawki PR nie uruchomiono buildera, nie
wygenerowano klucza/podpisu, nie utworzono approval i nie zmieniono żadnego
istniejącego artefaktu, VM lub snapshotu.

## Obowiązujący kontrakt

Obecny format nie ma osobnych pól rewizji aplikacji i rewizji zależności
runtime. Zatwierdzone wydanie ma jeden pełny identyfikator Git `R`:

| Materiał lub bramka | Wymagana zgodność |
| --- | --- |
| Manifest runtime v1 | `release_revision == R`; `_schema` porównuje do `RuntimeConfig.expected_revision` |
| Chroniony binding runtime | `source_revision == release_revision`; `verify_installed_runtime` wybiera tę samą rewizję do weryfikacji podpisanego manifestu |
| Podpisany manifest aplikacji v2 | `source_revision == R` i `runtime.source_revision == R`; cały obiekt `runtime` odpowiada chronionemu bindingowi bez `schema` |
| Eksporter aplikacji v2 | `HEAD == R`, bloby i bajty źródeł pochodzą z `R`, `_binding` odmawia bindingu innej rewizji |
| Instalator | `--expected-revision == R`, manifest aplikacji i jego obiekt runtime wskazują `R`; zweryfikowany runtime również musi wskazywać `R` |
| Instalator i bootstrap | `freshness_runtime.py`, `freshness_runtime_bootstrap.py` i `freshness_application.py` w payloadzie odpowiadają dokładnym niezależnie zaufanym bajtom helperów instalatora |
| Kontrola zainstalowanej aplikacji | Zweryfikowany runtime, podpisany manifest aplikacji i jego obiekt runtime mają identyczną rewizję |
| Chronione drzewo harnessu i wykonanie | Zewnętrzny inwentarz `source_revision == R`, dokładne źródła z `R`, `--revision == R` |
| Zewnętrzne dopuszczenie VM | `revision == R` i ten sam `environment_id` oraz właściwe obserwacje hosta/bootu |

Źródła kontraktu: `deployment/freshness_runtime.py` (`_schema`,
`verify_installed_runtime`), `scripts/prepare_stage9_vm_bundle.py` (`_binding`,
`prepare`), `deployment/production_local_installer.py` (`_manifest`,
`_trusted_bootstrap`, `_verified_runtime`), `deployment/freshness_application.py`
(`verify_installed_application`, `_runtime_binding`) oraz `stage9_vm_trial.py`
(`_harness_sources`, `_installed_executable`). Weryfikacja podpisu nadal
poprzedza interpretację manifestu. Schemat lub funkcja równości nie jest
samodzielnym uwierzytelnieniem materiałów.

**Wniosek: wdrożenie nowego zatwierdzonego commita wymaga w obecnym kontrakcie
nowego wydania runtime, uzyskanego przez osobno zatwierdzoną budowę oraz
niezależną ocenę i podpisanie nowych materiałów.** Nie wystarczy zachować
stary runtime i zmienić wyłącznie rewizję manifestu albo bindingu.

Nie dowodzi to konieczności zmiany wersji zależności ani ich bajtów. Przy
niezmienionym locku część payloadu nowego runtime może pozostać identyczna.
Konieczność nowego wydania wynika z tożsamości podpisanych materiałów i równości
rewizji, a nie z wykazanej wady pakietu Ubuntu. Obecny builder jest dostępną
ścieżką utworzenia nowego kandydata; nie ma zatwierdzonej operacji przeniesienia
starego wydania na inną rewizję bez budowy.

W szczególności runtime z `fc15b98298c5430286dd2206ec7eeb8f20e2497b` nie
ustanawia zgodnego podłoża dla aplikacji i harnessu z
`5e0646a533d071b3f4a1c62d8d9836ece9081c95` ani dla kolejnego commita
poprawiającego PR. Zmieniony bootstrap należy do materiałów aplikacji v2 i do
niezależnej kotwicy instalatora. Przepisanie nowej aplikacji na stare `R` nie
udowodni pochodzenia jej nowych bajtów. Powrót do starego bootstrapu porzuciłby
remediację OpenSSL.

## Jawne niedozwolone kombinacje

- Manifest runtime `fc15…` + oczekiwany nowy commit: odmowa `_schema`.
- Binding `fc15…` + eksport aplikacji z nowego commita: odmowa przed pierwszym
  zapisem nowego eksportu.
- Manifest aplikacji nowy + jego obiekt runtime stary, niezależnie od tego,
  czy wskazany runtime stary czy nowy: odmowa.
- Runtime nowy + stary obiekt runtime w aplikacji lub stary binding: odmowa.
- Wszystkie lokalne pola stare, ale `--expected-revision` nowy: odmowa
  instalatora; jego bramka runtime porównuje tę rewizję także bezpośrednio.
- Payload helperów nowy + stara kotwica instalatora albo odwrotnie: odmowa
  porównania dokładnych bajtów, nawet gdy reszta pól jest spójna.
- Harness lub zewnętrzne dopuszczenie VM ze starej rewizji przy nowym `R`:
  odmowa; nie tworzyć syntetycznego dopuszczenia, żeby ją ominąć.

## Procedura operatora po uzyskaniu odrębnej zgody

Poniższe czynności są planem przyszłej migracji, nie poleceniem do wykonania
w aktualnym zadaniu. Każdy etap, którego warunki nie są spełnione, pozostaje
BLOCKED. Nie modyfikować istniejącej VM, snapshotu, runtime, manifestu, bindingu
ani podpisu w celu wykonania tej procedury.

1. **Zamrozić finalne `R`.** Po ponownym review ustalić pełny SHA finalnego
   commita PR, zatwierdzony zakres źródeł i wyniki dozwolonego CI właśnie dla
   niego. `R` nie jest automatycznie aktualnym `HEAD`, `main`, wcześniejszym
   `fc15…` ani wcześniejszym `5e064…`. Wszelkie późniejsze zmiany źródeł lub
   identyfikatora wykonywanego wydania wymagają ponownej oceny zgodności.
   Artefakty i decyzje przechowuje się poza źródłowym commitem, więc nie należy
   tworzyć cyklu „commit manifestu zmienia SHA, który manifest ma wskazywać”.
2. **Uzyskać osobne upoważnienie do dokładnych operacji.** Potrzebne są nowy
   kandydat runtime dla `R`, świeży `environment_id`, platforma Ubuntu 24.04
   amd64/CPython 3.12 i nowy odizolowany obszar budowania. Obecny builder
   wykonuje sondę `_PROBE`, która generuje efemeryczny klucz Ed25519 i podpis
   wiadomości. Zgoda musi obejmować także tę operację; obecny zakaz jej nie
   obejmuje wyjątkiem. Wymagane później podpisanie nowych manifestów i odrębna
   decyzja dotycząca nowego hosta/VM również wykraczają poza ten PR. Bez takiego
   upoważnienia nie uruchamiać buildera ani historycznych komend `genpkey`/`sign`
   z innych runbooków.
3. **Zachować stare materiały.** Zarchiwizować ich istniejące bajty, hashe,
   pochodzenie, logi, podpisy i powiązania jako materiał wydania `fc15…`, bez
   przepisywania manifestu. Nie oznaczać starej generacji jako zgodnej z nową.
   Nowe katalogi kandydatów, pliki manifestów i identyfikator środowiska muszą
   być nieużywane. Nigdy nie nadpisywać starego wydania lub śladu awarii.
4. **Uwierzytelnić narzędzia i zależności.** Sprawdzić dokładne źródła buildera,
   jego helperów, lock i katalog wheelhouse z `R`. Builder wymaga osobnego
   zaufanego zestawu narzędzi; minimalne 18 plików harnessu nie zawiera buildera.
   Zweryfikować wersje i pochodzenie Ubuntu, interpretera, loadera, libc,
   OpenSSL i natywnych bibliotek. Brakujące dowody ze
   [ścieżki proweniencji](stage9_runtime_provenance.md) pozostają BLOCKED i nie
   znikają po utworzeniu nowego kandydata. Nie aktualizować locka bez wykazanej
   potrzeby. Budowa ma działać jako nie-root, offline, z zatwierdzonych wheels.
5. **Zbudować tylko nowego kandydata, dopiero po upoważnieniu z punktu 2.**
   Przykładowe argv dla niezależnie uwierzytelnionego buildera; wszystkie
   zmienne są wartościami ustalonymi przez operatora, a ścieżki absolutne:

   ```sh
   /usr/bin/python3.12 -I -s -B "$S9_TRUSTED_BUILDER" \
     --wheelhouse "$S9_AUTHENTICATED_WHEELHOUSE" \
     --destination "$S9_NEW_RUNTIME_CANDIDATE" \
     --environment-id "$S9_NEW_ENVIRONMENT_ID" \
     --release-revision "$S9_APPROVED_COMMIT" \
     --manifest "$S9_NEW_RUNTIME_MANIFEST"
   ```

   Zachować argv, UTC, stdout/stderr i oryginalny exit, pełny inwentarz,
   `build-probe.json`, wersje, lock i hashe. Ten krok nie zmienia starego
   runtime, nie tworzy bindingu i nie jest kwalifikacją VM. Nie wykonywać
   samego `collect_runtime_manifest` na starym drzewie, żeby zmienić jego etykietę.
6. **Niezależnie ocenić nowy runtime.** Rola autorytetu sprawdza nowy manifest,
   cały inwentarz wraz z dodatkowymi obiektami, pochodzenie i wyniki sond.
   Dopiero odrębna upoważniona ceremonia może wytworzyć podpis nowych bajtów.
   Tu nie podaje się ani nie uruchamia komend generujących klucze/podpisy.
   Brak autentycznego wyniku tej ceremonii zatrzymuje kolejne etapy.
7. **Przygotować świeży docelowy host, wyłącznie po osobnym dopuszczeniu.**
   Użyć nowej przestrzeni TEST_ONLY, bez modyfikowania obecnej VM/snapshotów.
   Nowy runtime przechodzi odczytową weryfikację z `--revision R`, a dopiero
   później osobną publikację do nieistniejącej ścieżki nowego `environment_id`.
   Staging/journal, istniejący cel albo niezgodność powodują odmowę i ręczne
   rozliczenie; nie kasować ich automatycznie. Chroniony binding nowego hosta
   odtwarza się z niezależnie zweryfikowanych nowych materiałów. Zmiana
   bindingu starego hosta nie jest częścią tej procedury.
8. **Wyeksportować nową aplikację v2 z tego samego `R`.** Na czystym checkoutcie
   `R`, jako nie-root, podać nowy zweryfikowany binding oraz nieistniejący
   output. Eksporter materializuje unit z nową ścieżką interpretera przed
   hashami, obejmuje zmieniony bootstrap i zapisuje `source_revision=R`:

   ```sh
   python -I -s -B "$S9_TRUSTED_EXPORTER" \
     --source "$S9_CHECKOUT_AT_R" --revision "$S9_APPROVED_COMMIT" \
     --binding "$S9_NEW_VERIFIED_BINDING_COPY" --output "$S9_NEW_APP_CANDIDATE"
   ```

   Nie przenosić podpisu starej aplikacji ani hashy helperów. Oddzielnie
   uwierzytelnić bootstrap instalatora i pełne chronione drzewo harnessu z `R`
   według [procedury drzewa](stage9_protected_execution_tree.md). Transportowy
   TAR i rootless kandydat pozostają danymi, nie instalacją.
9. **Niezależnie ocenić aplikację i kotwicę instalatora.** Autorytet aplikacji
   pozostaje odrębny od runtime. Jakiekolwiek przyszłe podpisanie lub dopuszczenie
   należy do osobnej upoważnionej ceremonii. Nie tworzyć lokalnego modelu approval
   zamiast autentycznych materiałów. Zgodność rewizji jest konieczna, ale sama
   nie ustanawia zaufania ani dopuszczenia VM.
10. **Najpierw odczytowe bramki na nowym, dopuszczonym hoście.** Porównać
    wszystkie tożsamości z tabeli, wykonać weryfikację runtime i preflight
    instalatora z tym samym `R`, zachować stany przed/po i exity. Nie przechodzić
    do instalacji, startu systemd, PostgreSQL ani prób negatywnych z mutacją,
    zanim ich odrębne warunki nie zostaną spełnione. Nieprzeprowadzony etap to
    NOT_EXECUTED, nie PASS. Każde niejednoznaczne rozliczenie pozostaje BLOCKED.

W ramach aktualnej remediacji wykonano wyłącznie modele równości i odmowy
rewizji, odczyt źródeł bootstrapu oraz dozwolone testy. Nie wykonano żadnego
z mutujących etapów migracji. Nowe wydanie runtime i dalsza próba Stage 9
pozostają **BLOCKED do odrębnej decyzji operatora**.
