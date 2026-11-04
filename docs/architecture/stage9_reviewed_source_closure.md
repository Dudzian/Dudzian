# Stage 9 — rozdzielenie reviewed source closure i podpisanego payloadu

Baza: `main` `a05c2a33f7e6c5078ba7112a6beaf3ac4b5170bf`.
[Niezależny audyt](stage9_reviewed_source_audit.md) oraz
[pełne hashe i diff](stage9_reviewed_source_audit.json) są materiałem do decyzji
reviewera, bez zatwierdzenia nowych bajtów. Nie zmieniono żadnego z sześciu
wpisów `REVIEWED_CODE_DIGESTS`.

## Dwa odrębne zakresy

`qualify_signed_freshness_substrate()` oraz jawne CLI
`--profile SIGNED_FRESHNESS_SUBSTRATE` odtwarzają podpis aplikacji v2, wszystkie
zainstalowane bajty i tryby, pięć istniejących pinów minimalnej usługi oraz
niezależnie podpisany runtime. Wynik JSON zawsze wskazuje
`qualification_profile=SIGNED_FRESHNESS_SUBSTRATE` i
`production_qualification=WITHHELD`. Nie jest dowodem działającej usługi,
poprawności PostgreSQL, hosta, AccountGenesis ani pełnego produktu.

Domyślny profil `PRODUCTION_LOCAL`, funkcja pełnego kwalifikatora i konsumenci
`deployment.platforms.linux.qualify_live()` nadal wymagają wszystkich sześciu
pinów, ochrony konfiguracji, rzeczywistej tożsamości usługi i kwalifikacji bazy.
Pełny kwalifikator dodatkowo wymaga podpisanej aplikacji; nie zastępuje tej
bramki wynikiem minimalnym. Aktualne bajty AccountGenesis nie spełniają pinu,
a instalacja minimalna nie zawiera pełnego produktu. Pełny wynik pozostaje
**BLOCKED**. Żaden pozytywny test z modelem hosta nie otwiera tej kwalifikacji.

## Podpis, inwentarz i ponowny start

Minimalny literalny `REQUIRED_FILES` zawiera obecnie 16 ścieżek, w tym nowy
`deployment/freshness_application.py`; eksporter nadal odczytuje ten kontrakt
przez AST. Nie dodano klienta AccountGenesis do minimalnej usługi. Pięć
reviewed digestów pochodzi z zaufanego helpera aplikacji; test sprawdza ich
zgodność z odpowiednim podzbiorem pełnego kwalifikatora. Pozostałe pliki
wymagają dokładnego wpisu w podpisanym manifeście. Dodatkowe podpisane pliki
mogą rozszerzać inwentarz, a pliki bez podpisanego wpisu są odrzucane.

Instalator przechowuje dokładny manifest, podpis i kopię niezależnego klucza
publicznego w chronionym `/etc/cryptohunter/freshness-application/`. Publiczny
`binding.json` wiąże manifest, rewizję i runtime, a powstaje dopiero po
ukończeniu instalacji. Niekompletny zapis nie ustanawia zatwierdzenia.
Idempotentny odczyt porównuje również te zachowane materiały. Istniejące
wydanie, katalog zatwierdzenia lub niekompletna instalacja nie są adoptowane
ani nadpisywane. Ponowne sprawdzenie wejściowego snapshotu przed publikacją
odrzuca zmianę pliku, manifestu lub runtime po wewnętrznym preflight.

Launcher weryfikuje podpis i dokładny inwentarz ponownie przed pierwszym
importem `bot_core`. Odmawia brakujących/dodatkowych plików, podmiany bajtów,
niezgodnych trybów, symlinków, hardlinków i cache importów. Porównuje również
`sys.executable`, inode `/proc/self/exe` i SHA-256 uruchomionego interpretera
z podpisanym runtime. Skopiowanie modułu przez administratora nie jest
podpisanym zatwierdzeniem.

Bootstrap instalatora, `freshness_application.py`, helper runtime, bootstrap
Ed25519/OpenSSL i bazowy interpreter muszą pochodzić z osobno zatwierdzonego,
chronionego źródła. Kod tej bazy zaufania wykonuje się przed sprawdzeniem
własnych bajtów. Launcher ma zostać uwierzytelniony w `ExecStartPre`, zanim
systemd go wykona. Podpis payloadu sam nie uwierzytelnia kodu sprawdzającego podpis.
Root i superuser PostgreSQL pozostają poza gwarancją; nie deklarujemy
odporności na równoczesną złośliwą mutację dokonywaną przez root po kontroli.

### Ciągłość zaufanego bootstrapu po review PR #3247

Przegląd HEAD `8207c8326301b5f21d18af43f830b8b4a32e01fd` potwierdził lukę:
poprawnie podpisany manifest mógł zawierać inne bajty
`freshness_runtime_bootstrap.py`, `freshness_runtime.py` lub
`freshness_application.py` niż odrębnie zatwierdzone kopie instalatora.
Pięć pinów usługi tego nie wykrywało. Ochrona właściciela i trybu pliku
zapobiega zapisowi przez zwykłego użytkownika, ale nie dowodzi pochodzenia
już skopiowanych bajtów. Nie jest też niezależnym podpisem.

Remediacja wiąże te trzy istniejące pliki payloadu z dokładnymi bajtami
zaufanych siblingów instalatora przed publikacją. Źródłem porównania nie jest
payload ani hash wyprowadzony z niego. Zmiana bootstrapu wymaga nowego,
odrębnego review i przygotowania zaufanej kopii operatora. Nie dodaje się
nowego pinu, pliku zaufania ani elementu `REQUIRED_FILES`.

`ExecStartPre` wykonuje zaufany systemowy Python, bootstrap i core runtime;
następnie zaufany helper aplikacji odtwarza podpisany inwentarz aplikacji.
Dopiero sukces tej kontroli pozwala systemd wykonać podpisany interpreter
i launcher. Launcher powtarza kontrolę aplikacji i tożsamości uruchomionego
interpretera przed importem usługi. Podpisana podmiana launchera jest zmianą
autoryzowanego payloadu; nie może zostać wykonana przed weryfikacją tego
podpisu. Nadal wymaga niezależnego review dokładnego manifestu wydania.

Niezależna kotwica pozostaje poza tym łańcuchem: operator odrębnym kanałem
uzyskuje fingerprint autorytetu, przeprowadza review dokładnej rewizji,
weryfikuje podpis `bootstrap.sha256` zaufanym systemowym OpenSSL oraz wszystkie
wymienione hashe przed uruchomieniem Pythona z bootstrapu. Po skopiowaniu do
chronionej lokalizacji ponawia porównanie bajtów i zapisuje metadane oraz
wynik kontroli. Eksporter tylko tworzy niepodpisanego kandydata. Plik sum
z tego samego, nieznanego bundle, samo approval VM lub sam `root:root/0644`
nie ustanawiają autentyczności. Kod instalacji i startu nie weryfikuje własnej
autentyczności przez samohash; polega na tej zewnętrznej ceremonii i ochronie
już zatwierdzonej kopii. Root, systemowy interpreter, OpenSSL i administrator
autorytetu pozostają zaufaną granicą.

Jawny test kontraktowy wymaga dokładnej równości
`freshness_application.REQUIRED_FILES == production_local_installer.REQUIRED_FILES`.
Testy podpisanych podmian helperów muszą odmówić przed zapisem. Test pomostowy
instalacji i launchera sprawdza rzeczywisty podpis aplikacji, zachowany binding,
import zainstalowanego kodu i konstruktory FreshnessVerifier, ale modeluje
runtime i granice hosta/bazy/socketu. Nie stanowi dowodu natywnego startu
systemd ani poprawnego żądania obsłużonego przez produkcyjny runtime.
Dokładne kryteria przyszłej próby określa
[kontrakt ACCEPT/REJECT izolowanej VM](../runbooks/STAGE9_FRESHNESS_SERVICE_VM_ACCEPTANCE.md).

### Oddzielna decyzja dotycząca AccountGenesis

Wszystkie sześć `REVIEWED_CODE_DIGESTS`, w tym obecny pin AccountGenesis,
pozostaje zamrożone. Przed zaakceptowaniem nowych bajtów niezależny reviewer
musi wskazać dokładną rewizję i digest, odczytać audyt bajtowy i semantyczny,
ocenić pełną ścieżkę IPC/PostgreSQL oraz odmowy, a następnie zapisać odrębną
decyzję autoryzacyjną. Hash aktualnego checkoutu i sukces CI nie zastępują
tej decyzji. Dopiero odrębnie zatwierdzone zmiany mogą prowadzić do nowego
eksportu i podpisu wydania. Ten PR nie aktualizuje tego pinu.

Stan Stage 9 pozostaje **IN_PROGRESS / PRODUCTION_NO_GO**. Pełna kwalifikacja
AccountGenesis, rzeczywiste wydanie, żywe dowody oraz uruchomienie produktu
pozostają poza remediacją tego review.

## Eksporter i zgodność wydania

Commit bazowy `4d42b1e1` już stosował początkowe `0600`, lecz podnosił tryb
przed opróżnieniem bufora. Eksporter teraz sprawdza pełną długość zapisu,
wykonuje `flush`, `fsync`, a dopiero potem ustawia wymagany tryb końcowy
`0644`/`0755`. Błąd pozostawia niezatwierdzony materiał w restrykcyjnym trybie;
ponowny zapis nie nadpisuje istniejącego pliku. Nie dodano suppressions.

Dotychczasowy podpisany manifest v2 bez nowego helpera nie spełnia nowego
inwentarza. Wymagany jest nowy eksport i osobny autoryzowany podpis po review.
Nie modyfikowano historycznych podpisów ani dowodów TEST_ONLY, nie używano
produkcyjnych kluczy. Dlatego domknięcie **rzeczywistego wydania** pozostaje
`SIGNED_PAYLOAD_CLOSURE=BLOCKED`, mimo pozytywnych testów kontraktu.

Testy negatywne obejmują SHA-256, zmodyfikowany AccountGenesis, brakujące i
dodatkowe pliki, rewizję spoza oczekiwanego wydania, manifest i podpis,
podmianę zakresu minimalnego na pełny, zmianę po preflight, podmianę po
instalacji, kolizje eksportera, przerwany zapis i restrykcyjny umask.
Istniejące progi, zakresy i manualne bramki fizyczne pozostają aktywne.
Nie wykonywano instalacji fizycznej, tworzenia kont, klastrów, enrollment,
restartu, Stage 10 ani merge. Natywna próba Linux VM pozostaje **PENDING**.
