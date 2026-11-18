# Stage 9 V5d4 — przygotowanie środowiska kontrolera

Uzupełnienie dla przypiętego wheel PySide6-Essentials 6.10.3:
[polityka dwóch pominięć bytecode i raport kolizji](stage9_v5d4_wheel_bytecode_remediation.md).
Pełny `RECORD` poprzedza decyzję o pominięciu; obowiązuje osobna, niezależna
akceptacja wyjątku kontraktu instalacji. Bytecode nigdy nie trafia do celu,
a 62 kolizje Essentials/Addons nadal powodują odmowę.

Baza: `913ae53a2e351291f86b05c4e3de1bb56c3559e1`, repozytorium
`Dudzian/Dudzian`. Narzędzia budują nowy lokalny katalog z zatwierdzonych
wejść; nie instalują do `/opt/cryptohunter` ani `/etc/cryptohunter`.
Nie tworzą podpisów, kluczy ani niezależnej autoryzacji.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```

## Rozwiązanie i granice dowodów

W V5d3 brakowało przygotowania rzeczywistych bajtów środowiska. V5d4 dodaje:

| Narzędzie | Wynik |
| --- | --- |
| `scripts/stage9_v5d4_closure.py` | Graf AST z inicjalizatorami, fazą importu, literałami dynamicznymi, nierozwiązanymi mechanizmami i hashem kontraktu instalacji. |
| `scripts/stage9_v5d4_dependencies.py` | Propozycja dokładnego domknięcia wheels, metadanych, SHA-256 i natywnych członków; osobna kontrola wobec dostarczonych oczekiwań. |
| `scripts/stage9_v5d4_abi.py` | Statyczna inspekcja CPython/libvirt/importów i pełnego jawnego grafu ELF przez `readelf`, bez `ldd` i bez uruchamiania badanych plików. |
| `scripts/stage9_v5d4_environment.py` | Offline staging venv bez pip i systemowych site-packages, plików kontrolera, wheels i osobno przypiętego bindingu libvirt. |
| `scripts/stage9_v5d4_launch.py` | Start z `-I -S -B`, bez zmiany `sys.path`; jawna bramka native import i odrębna bramka odczytowego V5d1. |

[Analiza zależności](stage9_v5d4_dependency_analysis.md) rozdziela 27 źródeł
osiągalnych przez kontroler od pełnych 26 deklaracji `project.dependencies`
i `defusedxml==0.7.1` z extra `observer-kvm-preflight`. Domknięcie metadanych
pełnej aplikacji pozostaje obowiązkowe przy budowie. Nie zastępuj go dwiema
bibliotekami bezpośrednio używanymi przez preflight. Opcjonalny dekompresor
`zstandard` jest osiągalny przez wspólne moduły, lecz nie jest wywoływany
przez zwykły V5d1; aktywacja offline-verifier wymaga odrębnego profilu i review.

Informacje operatora o Ubuntu 24.04.5, x86_64, CPython 3.12.3, SOABI
`cpython-312-x86_64-linux-gnu`, python3-libvirt/libvirt 10.0.0 i QEMU 8.2.2
są wymaganiami wejściowymi. Nie były ponownie mierzone na fizycznym hoście.
Wcześniejszy `.venv` CPython 3.11 nie spełnia tego profilu; kopiowanie
`libvirtmod.cpython-312-…so` do niego nie rozwiązuje ABI.

## Analiza bezpieczeństwa nowego sposobu startu

V5d1 i jego podpisany kontrakt pozostają bez zmian. Nowy katalog jest propozycją
do przyszłej instalacji, a nie chronioną instalacją lub nowym sposobem admission.
Nie jest dopisywany do istniejącej polityki. Dotychczasowe bramki tożsamości,
niezależnego launch proof, chronionych plików i blokady wykonania nadal obowiązują.

Zwykłe uruchomienie venv wykonuje `site` przed importem kontrolera. Projektowy
wheel zawiera również `sitecustomize`, a pakietowanie projektu nie obejmuje
`deployment`; dlatego ten profil nie wykonuje `pip install .` ani startupu
takiego wheel. Staging przenosi jawne, oddzielnie przypięte źródła kontrolera
wraz z inicjalizatorami i pełne zależności z dotychczasowego kontraktu.
Ta różnica została jawnie wyodrębniona: nie rozszerza zaufania do kodu aplikacji
ani nie zmniejsza zestawu jej deklarowanych zależności.

Launcher uruchamia się wyłącznie z izolacją, bez `site` i bez zapisu bytecode cache.
Nie czyta `.pth`, `sitecustomize`, `usercustomize` ani `PYTHONPATH`.
Pakiety rozwiązuje dodatkowy `MetaPathFinder` ze stałego `packages/`, po
standardowych finderach stdlib. Nie zmienia globalnego `sys.path`.
Pakietowe `__path__` i wykonanie inicjalizatorów nadal wymagają review:
ten mechanizm nie jest sandboxem dla złośliwych zatwierdzonych pakietów.

Binding jest kopiowany jako dokładne pliki `libvirt.py` oraz
`libvirtmod.cpython-312-x86_64-linux-gnu.so`; nie udostępnia całego
`/usr/lib/python3/dist-packages`. Biblioteki systemowe pozostają zewnętrznymi,
osobno przypiętymi wejściami ELF. Wszystkie natywne pliki wheels, także prywatne
ELF, muszą znaleźć się w pomiarach ABI. Kolizje plików, `.pth`, cache, startup
customization, dowiązania i nieobsługiwane układy wheel są odmową.

Lokalne hashe dowodzą zgodności z oczekiwaniami, nie pochodzenia oczekiwań.
Hash przekazany przez recenzenta musi być uzyskany niezależnym kanałem.
Narzędzia nie uwierzytelniają recenzenta ani nie zastępują ceremonii V5d1.
Sprawdzenie już uruchomionego procesu nie ustanawia zaufania do jego startu.
Interpreter bazowy, jego stdlib, loader, narzędzia oraz pochodzenie pakietów
systemowych muszą zostać osobno zaakceptowane przed uruchomieniem narzędzi.
Przypięte bajty header/sysconfigdata nie dowodzą same związku z konkretnym
ELF interpretera: ten związek wymaga dowodów dystrybucji/builda recenzenta.

## Procedura przygotowania

Wszystkie poniższe polecenia dotyczą przyszłego, niezależnie przejrzanego
hosta przygotowującego. Nie uruchamiaj interpretera z badanego obrazu gościa.
Nie używaj `sudo`, `activate` ani skryptów z pobranych dystrybucji.
Przykłady używają zwykłych katalogów użytkownika; zastąp nazwy rzeczywistymi.

1. Uzyskaj przejrzaną kopię narzędzi i źródeł dokładnej rewizji. Użyj
   zaufanego systemowego `/usr/bin/python3.12`, a nie dawnego `.venv` 3.11.
   Zapisz graf źródeł jako dane:

   ```bash
   /usr/bin/python3.12 -I -S -B scripts/stage9_v5d4_closure.py \
     --repository /home/operator/reviewed-source --output /home/operator/closure.json
   ```

2. Osobny proces pozyskania/resolver przygotowuje wheelhouse dla CPython
   3.12/Linux x86_64/glibc 2.39, z pełnych deklaracji projektu i extra.
   Użyj wyłącznie wheels, bez buildów SDist i hooks. W katalogu musi być
   dokładnie jeden wybrany wheel na każdą osiągalną nazwę. Pozyskanie nie
   oznacza aprobaty. Zapisz `origins.json`, mapę nazw plików na obiekty:

   ```json
   {"nazwa-wersja-py3-none-any.whl": {"url": "https://niezaleznie-sprawdzone-zrodlo/plik.whl", "source": "referencja procesu pozyskania", "sha256": "64 male znaki hex"}}
   ```

   URL/source opisują propozycję pochodzenia. Recenzent porównuje je z
   niezależnie uzyskanym indeksem/archiwum i dowodami dostawcy.

   ```bash
   /usr/bin/python3.12 -I -S -B scripts/stage9_v5d4_dependencies.py propose \
     --pyproject /home/operator/reviewed-source/pyproject.toml \
     --wheelhouse /home/operator/wheelhouse --origins /home/operator/origins.json \
     > /home/operator/dependency-proposal.json
   ```

   Kod 0 oznacza kompletność obsługiwanego domknięcia metadanych, nie zatwierdzenie.
   Narzędzie sprawdza każdą wersję, marker i extra, `Requires-Python`, tagi wheel,
   zgodność nazwy/metadanych, SHA całej dystrybucji i każdego członka `RECORD`.
   Nie jest resolverem z backtrackingiem. Obsługuje finalne wersje numeryczne
   i `.post`, typowe operatory PEP 440 i PEP 508 (`and`, `or`, extras, `in`).
   Pre/dev/epoch/local i nieobsługiwane markery odmawiają zamiast zgadywania.

3. Recenzent, używając oddzielnie przejrzanego weryfikatora, porównuje
   propozycję, kontrakt, źródła, pochodzenie i natywne bajty. Oczekiwany hash
   propozycji to SHA-256 kanonicznego JSON (`sort_keys=True`, separatory
   `,` i `:`, bez końcowego LF); hash pliku approval jest hashem surowych bajtów.
   Nie pobieraj oczekiwanego hasha z samej propozycji jako substytutu aprobaty.

   ```bash
   /usr/bin/python3.12 -I -S -B scripts/stage9_v5d4_dependencies.py verify \
     --pyproject /home/operator/reviewed-source/pyproject.toml \
     --wheelhouse /home/operator/wheelhouse --origins /home/operator/origins.json \
     --proposal /home/operator/dependency-proposal.json \
     --expected-sha256 "$proposal_digest_od_recenzenta" \
     --approved-native /home/operator/reviewed-native.json
   ```

   `reviewed-native.json` jest dokładną mapą `wheel.whl:ścieżka/członka` na
   SHA-256. Każdy dodatkowy lub brakujący natywny członek daje odmowę.

   Osobno zinwentaryzuj bazowy runtime CPython, używając `--base-runtime-root`
   tego samego generatora closure. Podaj wszystkie istniejące ścieżki
   bazowego `sys.path` uruchomienia `-I -S`, także zip, jeśli istnieje.
   Nadkatalog stdlib obejmuje jego `lib-dynload`:

   ```bash
   /usr/bin/python3.12 -I -S -B scripts/stage9_v5d4_closure.py \
     --base-runtime-root /usr/lib/python3.12 --output /home/operator/base-runtime.json
   ```

   Ten konserwatywny nadzbiór obejmuje również cache `.pyc` i fizyczne cele
   jawnie zapisanych dowiązań. `-B` nie zabrania odczytu istniejącego cache.
   Wykaz jest propozycją: recenzent porównuje wszystkie bajty i ogniwa
   dowiązań z bazowymi pakietami dystrybucji/builda. `site-packages` i
   `dist-packages` są pomijane, ponieważ launcher nie włącza ich przez `site`.
   Builder porównuje cały wykaz z `approval.base_runtime` oraz wymaga objęcia
   każdej istniejącej ścieżki importów. Launcher ponawia inwentaryzację
   zatwierdzoną kopią `runtime-audit.py`, wykrywając również dodatkowe pliki.

4. Przygotuj request ABI i oczekiwania zgodnie z
   [procedurą statyczną](stage9_v5d4_static_abi.md). Dla buildera ścieżki
   odnoszą się do przyszłego, nieistniejącego katalogu wynikowego:
   `interpreter=<wynik>/venv/bin/python3.12`, jedyny `import_roots` to
   `<wynik>/packages`, bindingi wskazują kopie w tym katalogu.
   `additional_extensions` musi obejmować każdy natywny plik z wheels,
   całego bazowego runtime (fizyczne ścieżki `base_runtime.native`) oraz
   `/usr/bin/qemu-img`. Są to tylko dane dla readelf, żaden plik nie jest wykonywany.
   `required_imports` obejmuje co najmniej `libvirt`, `libvirtmod`,
   `cryptography`, `defusedxml`, a review uzupełnia pełny wymagany zakres.
   `pyvenv_cfg` wskazuje `<wynik>/venv/pyvenv.cfg`. Header, sysconfigdata,
   biblioteki i loader mają osobne, zatwierdzone ścieżki i hashe.
   Wyliczone pliki venv i docelowe ścieżki wymagają powtarzalnego review;
   nie dopisuj hashy brakujących dowodów automatycznie jako akceptacji.

5. `inputs.json` zawiera dokładnie ścieżki danych używanych przez builder:

   ```json
   {"repository":"/home/operator/reviewed-source","wheelhouse":"/home/operator/wheelhouse","origins":"/home/operator/origins.json","proposal":"/home/operator/dependency-proposal.json","abi_request":"/home/operator/abi-request.json","abi_expectations":"/home/operator/abi-expectations.json"}
   ```

   Zewnętrzne `approval.json` zawiera pola `host`, `proposal_sha256`,
   `wheel_native`, `code`, `dynamic_review_sha256`, `abi_request_sha256`,
   `abi_expectations_sha256`, `bindings`, `base_runtime`, `host_tools`.
   `base_runtime` to cały niezależnie przyjęty wykaz opisany wyżej,
   `host_tools` to dokładna mapa `/usr/bin/qemu-img` na zatwierdzony SHA-256.
   Wersja i pochodzenie QEMU/libvirt wymagają zewnętrznego porównania pakietów
   systemowych; narzędzie nie uruchamia `--version` tych plików.
   To wejście recenzenta; narzędzie
   nie generuje aprobaty. `code` to dokładna mapa względnych paths z
   `closure.sources` na SHA. `dynamic_review_sha256` to hash kanonicznego
   pełnego raportu closure, z decyzją o używanych ścieżkach po stronie review.
   `host` ma dokładnie `python_full_version=3.12.3`, `implementation=CPython`,
   `machine=x86_64`, `system=Linux`, powyższe `soabi` i
   `interpreter_sha256` rzeczywistego hostowego ELF oraz `distribution=ubuntu`,
   `distribution_version=24.04`, `glibc=glibc 2.39`. `bindings` ma dokładnie
   dwa klucze nazw plików bindingu; każda wartość to `path` oryginalnego
   systemowego pliku i `sha256` jego zaakceptowanych bajtów.

   ```bash
   /usr/bin/python3.12 -I -S -B scripts/stage9_v5d4_environment.py \
     --inputs /home/operator/inputs.json --approval /home/operator/approval.json \
     --approval-sha256 "$approval_digest_od_recenzenta" \
     --output /home/operator/observer-prepared
   ```

   Builder wykonuje porównania zależności i kodu, tworzy venv bez pip, kopiuje
   wheels/binding, sprawdza ABI tych docelowych bajtów i dopiero wtedy zamyka
   `ENVIRONMENT.json`. Nie uruchamia nowego interpretera ani kontrolera.
   Częściowy wynik po odmowie nie ma poprawnej pieczęci i nie nadaje się do
   startu. Ponowienie wymaga nowego katalogu; nie nadpisuj poprzedniego.
   Środowisko nie jest relokowalne bez ponownego review ścieżek i polityki.

Obsługiwane są zwykłe wheels i relocation `.data/purelib`, `.data/platlib`.
Skrypty/prefix data, natywne pliki wymagające `.data` relocation, duplikaty
destynacji i zależności wymagające startup hooks wymagają osobnej analizy;
narzędzie odmawia takiego profilu. Nie wykonuje automatycznych obejść.

## Oddzielne bramki wykonania — nie wykonywane w tym zadaniu

Przed startem niezależnie sprawdź interpreter, stdlib, launcher i receipt.
Nie traktuj hasha wypisanego przez ten sam builder jako aprobaty.
Zwykłe uruchomienie launchera bez flag odmawia przed importem pakietów.
Wyłącznie po osobnej jawnej autoryzacji native load można użyć:

```bash
env -i LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC PATH=/usr/bin:/bin \
  /home/operator/observer-prepared/venv/bin/python3.12 -I -S -B \
  /home/operator/observer-prepared/launch.py \
  --authorize-native-load --receipt-sha256 "$receipt_digest_od_recenzenta"
```

Ten tryb importuje binding, bez wywoływania metod libvirt, pobierania wersji
od daemonu ani lifecycle VM. Kod 0 oznacza tylko zakończony import natywny.
Odrębny `--authorize-readonly-preflight` dodany na końcu powyższej komendy
wywoła istniejące `main([])` V5d1 i może dokonać jego odczytów libvirt/obrazów;
wymaga osobnego zatwierdzenia odczytowego preflight i wszystkich dotychczasowych
warunków V5d1. Nie wykonywano żadnego z tych trybów w tej pracy.
Prawidłowy V5d1 nadal zwraca kod 2 i blokadę wykonania.

Launcher porównuje cały lokalny wykaz plików oraz przypięte wejścia zewnętrzne.
To kontrola snapshotu: nie usuwa wyścigu po odczycie, nie ustanawia immutable
view ani dowodu tożsamości już załadowanego procesu. Dynamiczne `dlopen`, NSS,
pluginy libvirt/Qt, konfiguracja loadera i pełny rzeczywisty runtime closure
pozostają przedmiotem osobnego pomiaru i review. `STATIC_ABI_MATCH` nie
poświadcza rzeczywistego wyboru `ld.so` ani nie kwalifikuje hosta.

## Przekazanie archiwum V5d3

Naprawione wykrywanie AST nie klasyfikuje `collector.execute(...)` jako
dynamicznego importu. `exec`, `eval`, `__import__`, importlib, runpy, ctypes,
aliasy, `getattr` i odwołania do builtins pozostają raportowane; inicjalizatory
są analizowane tak samo jak pozostały kod.

Archiwizuj cały katalog próby: `bundle/`, `HANDOFF.json`, oryginalny stdout
`verification.json`, `verification.stderr`, `verification.exit`. Koperta wiąże
ich rozmiary i SHA z zamkniętymi manifestami, bez cyklicznego haszowania raportu
w jego własnym bundle. Zachowaj również surowy stdout/stderr i exit ponownej
weryfikacji odbiorcy poza archiwum. Nie zastępuj ich przepisanym podsumowaniem.

```bash
bash scripts/stage9_v5d3_verify_bundle.sh --handoff /home/operator/stage9-v5d3-proba
```

Brak raportu, kodu, checksumy lub zgodności daje odmowę. Samo sprawdzenie
`bundle/` potwierdza jego spójność, ale nie kompletność przekazania.
Resume zachowuje poprzednie oryginalne raporty w nowej próbie.

## Pozostałe blokady

Nie dostarczono rzeczywistego wheelhouse, zatwierdzonych oczekiwań ani
niezależnych dowodów pochodzenia hostowego runtime/systemowych bibliotek.
Nie stworzono więc fałszywego zaakceptowanego locka lub instalacji dla hosta
operatora. W tym środowisku zaufany Python ma 3.12.14, a nie wymagane 3.12.3;
rzeczywista budowa docelowego profilu musi odmówić. Testy stagingu używają
oznaczonych syntetycznych danych i zastępują granicę hosta/ELF.

Przyszła instalacja wymaga niezależnego review pełnej propozycji, base-runtime
provenance, immutable view, chronionego launchera, polityki V5d1, osobnej VM,
wymagań domeny/obrazów, kwalifikacji oraz osobnej autoryzacji eksperymentu.
Żaden wynik tego PR nie przyznaje `HOST_PREFLIGHT_VERIFIED`, tożsamości
kontrolera ani produkcyjnego admission.
