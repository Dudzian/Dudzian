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

Bootstrap instalatora, entrypoint, `freshness_application.py`, helper runtime,
bootstrap Ed25519/OpenSSL i bazowy interpreter muszą pochodzić z osobno
zatwierdzonego, chronionego źródła. Wykonują się przed sprawdzeniem własnych
bajtów. Podpis payloadu sam nie uwierzytelnia kodu sprawdzającego podpis.
Root i superuser PostgreSQL pozostają poza gwarancją; nie deklarujemy
odporności na równoczesną złośliwą mutację dokonywaną przez root po kontroli.

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
