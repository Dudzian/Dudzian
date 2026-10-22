# Stage 9 — deterministyczne HWID izolowanego custody Linux

Stage 9 pozostaje **NO-GO**. Ta zmiana usuwa zależność lokalnego custody od
losowego fallbacku `uuid.getnode()`. Nie kwalifikuje produkcyjnych usług,
wyłącznej kontroli klucza ani emisji Root Proof. Nie uruchamia provisioningu,
enrollmentu, BIND, SIGN ani Stage 10.

## Problem i granica zmiany

Fizyczne testy Ubuntu 24.04.4 wykazały stabilny fingerprint hosta w czterech
niezależnych procesach oraz niestabilność przy DynamicUser i
`RestrictAddressFamilies=AF_UNIX`, zarówno z `PrivateNetwork=no`, jak i `yes`.
`cpu_id`, `hostname` i `os` pozostawały stabilne; zmieniało się
`mac_addresses`, a `uuid.getnode()` wykazywało losowy fallback. Odczyt MAC
z `/sys/class/net` był stabilny, lecz w prywatnej sieci nie było niezerowego
adresu MAC. Stabilność kolekcji hosta nie dowodzi stabilności w signerze.

Zmiana kolekcji MAC w globalnym `fingerprint.py` zmieniłaby HWID używany przez
licencje i istniejące zaszyfrowane sekrety. Dlatego globalny fingerprint,
`HwIdProvider`, format licencji i domyślny magazyn sekretów zachowują swoje
dotychczasowe znaczenie. Linux native custody ról root-proof signing oraz
history-attestation signing (`_DB_FIRST_ROLES`) otrzymuje odrębną, jawnie
wersjonowaną tożsamość. Lokalny reader i administrator tych ról wybierają
schemat wewnętrznie. Role freshness/CHA i pozostałe platformy zachowują
dotychczasową ścieżkę; ten dokument nie kwalifikuje ich dla nowego schematu.

## Schemat `linux-host-v1`

`bot_core/security/custody_hwid.py` wymaga dwóch lokalnych składników:
identyfikatora instalacji `/etc/machine-id` i sprzętowego DMI product UUID.
Sam machine ID nie wystarcza: skopiowanie go wraz z keyringiem na inny sprzęt
nie może zachować powiązania custody. Brak jednego składnika powoduje odmowę;
nie ma automatycznego wyboru słabszego zestawu identyfikatorów.

Offline administrator odczytuje sprzętowy składnik bezpośrednio ze stałej
ścieżki `/sys/devices/virtual/dmi/id/product_uuid`. Zainstalowany signer
wybiera wewnętrznie przypisany swojej roli credential systemd, ponieważ
uprawnienia zwykle blokują bezpośredni odczyt DMI przez jego UID. Systemd
dostarcza zawartość źródła kernela przez:

```ini
LoadCredential=custody-product-uuid:/sys/devices/virtual/dmi/id/product_uuid
```

Obie dostarczone jednostki wymagają tej dyrektywy. Wybór portu jest jawny
i wewnętrzny, bez automatycznej próby drugiego źródła przy odmowie odczytu.
Rootowy manager odczytuje
źródło kernela i udostępnia je danej usłudze przez chroniony, read-only
credential mount. Nie jest to przyjęcie HWID podanego przez requestera ani
administracyjnego pliku zawierającego dowolną wartość. Provider dopuszcza
wyłącznie dwa stałe cele:

- `/run/credentials/cryptohunter-root-proof-signer.service/custody-product-uuid`
- `/run/credentials/cryptohunter-history-attestation-signer.service/custody-product-uuid`

Nie odczytuje `CREDENTIALS_DIRECTORY` ani innych zmiennych jako selektora
źródła. Nie przyjmuje wartości HWID ani ścieżki z IPC, CLI, konfiguracji
custody lub pliku requestera. Użytkownik nie dostaje prawa zapisu do
credential. Nie trzeba udostępniać hostowej sieci, katalogu home ani nowych
capabilities. Istniejące `ProtectSystem=strict`, `ProtectHome=yes`,
`PrivateNetwork=yes` i `RestrictAddressFamilies=AF_UNIX` pozostają aktywne.

Odczyt wymaga chronionego root-owned źródła i ancestry, odmawia podążania za
symlinkiem i akceptowania zapisu przez group/world oraz sprawdza typ i
ograniczony rozmiar pliku. Dla systemd credential katalog per-unit i plik
mogą należeć do root lub bieżącego UID signera, muszą mieć readonly mount
i nie mogą mieć żadnego bitu zapisu; `/run` i `/run/credentials` pozostają
chronioną root-owned ancestry. Dane muszą być kanonicznym, niezerowym
machine ID Linux oraz poprawnym, niezerowym sprzętowym UUID. Brak źródła,
błąd odczytu, zmiana podczas odczytu lub nieprawidłowa wartość powodują
odmowę. Nie ma fallbacku do MAC, hostname, losowego UUID, pustej wartości ani
nowo wygenerowanego identyfikatora. Hosty bez DMI, w tym platformy ARM bez
tego źródła, są nieobsługiwane i odmawiają; wymagają osobno przejrzanego
schematu zamiast cichego pominięcia składnika sprzętowego.

Wynik jest domain-separated i ma postać
`custody:linux-host-v1:<sha384-hex>`. Nazwa schematu i prefiks są częścią
kontraktu. Dodatkowe haszowanie HWID przez istniejący magazyn nadal wiąże
rekord klucza głównego i klucz AES-GCM z lokalną tożsamością. Wersja schematu
HWID jest niezależna od istniejącej wersji formatu zaszyfrowanego sekretu.
Obie koperty custody wymagają jawnego `hwid_scheme=linux-host-v1`
i zgodności digestu; niezgodna, brakująca lub nieznana wersja odmawia.

Runtime nadal odczytuje istniejący sekret oraz istniejący master przez
dokładną klasę natywnego `Keyring`. Chained backend, fallback,
brak mastera lub niezgodność HWID nie tworzą klucza, indeksu ani nowego
sekretu i nie uruchamiają migracji. Odrębne UID/GID i namespaces, chronione
ścieżki, role/service namespaces i weryfikacja seed/public metadata nadal
obowiązują. Jedyna zmiana jednostek to załadowanie obowiązkowego sprzętowego
składnika; ustawienia izolacji pozostają bez zmian.

## Zapis wymaga istniejącego mastera

W trybie `custody_hwid=True` zwykły `set_secret()` nie inicjalizuje magazynu.
Każdy zapis, także pod nową nazwą, wymaga istniejącego mastera z poprawnym
`hwid_scheme=linux-host-v1`, wersją oraz digestem bieżącego HWID. Przy
aktualizacji istniejącego sekretu trzeba dodatkowo poprawnie odszyfrować
jego poprzednią kopertę. Brak mastera powoduje odmowę przed generowaniem
klucza lub nonce, szyfrowaniem, zapisem ciphertextu i utworzeniem indeksu.
Wewnętrzny loader również zabrania tworzenia mastera custody niezależnie od
wartości parametru `allow_create`; nie można ominąć reguły innym wywołaniem
loadera. Domyślny tryb legacy zachowuje dotychczasowe zachowanie.

Nieobecność żądanego sekretu nie dowodzi, że native keyring jest pusty.
Może zawierać `old-seed`, którego master utracono, nawet jeśli zapisywany
`new-seed` jeszcze nie istnieje. Nowy master osierociłby stare ciphertexty
i nadał niespójne znaczenie tej samej kolekcji. Brakujący, pusty lub
nieaktualny plik indeksu nie jest dowodem pustej kolekcji i nie przyznaje
prawa do inicjalizacji, naprawy ani resetu. Odmowa zachowuje dotychczasowe
rekordy i indeks bez zmian; brakujący indeks pozostaje nieutworzony.

Pierwsza inicjalizacja wymaga odrębnej, jawnie autoryzowanej ceremonii
administracyjnej. Ten PR nie implementuje takiej ścieżki ani nie nadaje
uprawnień do niej. Dotychczasowy administrator lokalnego custody może
zapisywać sekrety tylko przy już istniejącym, zweryfikowanym masterze;
pierwszy provisioning bez niego pozostaje zablokowany. Przyszła procedura
wymaga osobnego przeglądu autoryzacji, dokładnego natywnego backendu,
weryfikacji właściwego urządzenia, roli i trust domain oraz wiarygodnego
ustalenia stanu kolekcji niezależnie od indeksu. Musi zapobiegać nadpisaniu
istniejącego mastera i wyścigom z innymi operacjami, zachować spójność
rekordów, chronione kopie, audyt bez ujawniania kluczy oraz obowiązujące
fences i izolację. Nie może być zwykłym przełącznikiem CLI, zmienną
środowiskową ani automatycznym skutkiem błędu odczytu.

Utrata mastera wymaga odrębnej procedury odzyskania oryginalnego klucza
z zaufanej, chronionej kopii przy zachowaniu powiązania z urządzeniem
i poświadczeniami. Nie wolno zastępować go losowo wygenerowanym masterem,
usuwać osieroconych danych ani oznaczać ich nowym schematem jako naprawy.
Brak bezpiecznej ścieżki odzyskania pozostawia custody zablokowane.

## Zgodność istniejących sekretów

Dotychczasowe rekordy ogólnego magazynu, licencje, role freshness/CHA
i pozostałe moduły nadal korzystają z dotychczasowego HWID. Nie są
przepisywane ani oznaczane nowym schematem. Istniejący Linux root-proof lub
history-attestation signing custody bez oznaczenia nowego schematu
odmawia użycia przez nową ścieżkę custody i wymaga kontrolowanej migracji.
Jest to jawna zmiana dostępności starego custody, a nie cicha zmiana znaczenia
jego klucza szyfrującego. Zmiana digestu lub samego oznaczenia schematu nie
jest migracją i nie zapewni poprawnego odszyfrowania.

Ten PR nie dostarcza narzędzia migracji ani nie wykonuje migracji. Wymaga ona
osobno przejrzanej procedury z następującymi warunkami:

1. Zatrzymać użycie obu wersji custody i uzyskać odrębne upoważnienie
   operacyjne. Zachować oryginalne rekordy mastera, wszystkie zaszyfrowane
   sekrety i odpowiadające im metadata w chronionej kopii; nie wypisywać
   seedów ani mastera do logów.
2. Na tym samym zweryfikowanym urządzeniu odtworzyć oryginalną zaufaną
   ścieżkę native backend i lokalnego legacy fingerprintu. Udowodnić
   zgodność starego HWID oraz poprawne uwierzytelnione odszyfrowanie każdego
   migrowanego sekretu. Nie przyjmować podanego przez operatora/requestera
   starego HWID, samego digestu z rekordu ani losowego `uuid.getnode()` jako
   dowodu tożsamości. Jeżeli oryginalnego powiązania nie można zweryfikować,
   odmówić i zachować blokadę; nie resetować mastera.
3. Zweryfikować canonical credential identity, trust domain, rolę,
   service namespace, public key i lifecycle przeciw chronionym metadata
   oraz właściwemu niezależnemu authority. Zachować istniejący seed i
   tożsamość poświadczenia. Samo odczytanie klucza nie upoważnia do zmiany
   lifecycle, enrollmentu, BIND ani aktywacji.
4. Wyliczyć nowe HWID wyłącznie przez chroniony lokalny provider. Jawnie
   ponownie zaszyfrować komplet odpowiednich rekordów i oznaczyć obie
   koperty schematem. Zapewnić spójny cutover mastera i wszystkich sekretów;
   częściowy zapis nie może zostać uznany za zakończoną migrację.
5. Przed cutover zweryfikować odczyt po niezależnym restarcie procesu przy
   docelowej izolacji i zgodność publicznego klucza. Udokumentować wynik,
   integralność kopii oraz kryteria rollbacku. Rollback przywraca razem
   oryginalny kod i cały spójny zestaw rekordów; nie łączy mastera jednej
   wersji z sekretami drugiej. Zachować NO-GO i pozostałe bramki produkcyjne.

## Zagrożenia i ograniczenia

Machine ID i DMI product UUID są publicznymi identyfikatorami. Nie są
sekretami ani sprzętowym lub TPM attestation. Hashowanie nie nadaje im tych
właściwości. Kopia obrazu wraz z machine ID i keyringiem na sprzęt o innym
DMI UUID odmawia; kopia całego wirtualnego sprzętu wraz z jego UUID może
odtworzyć tę samą tożsamość. Złamanie root, firmware, hypervisora, zaufanej
instalacji lub mount namespace może podmienić źródło. Te zagrożenia wymagają
niezależnej ochrony hosta, kontroli unikalności identyfikatorów obrazów/VM
oraz istniejącej ochrony natywnego magazynu. Nie wolno deklarować wyłącznej
kontroli klucza na podstawie HWID.

Ponowna instalacja hosta, regeneracja machine ID, zmiana DMI UUID lub
przywrócenie niespójnej kopii powodują odmowę. Nie ma automatycznej adaptacji
do nowego urządzenia.
Publiczny HWID i jego digest mogą korelować instalacje, dlatego nie należy
publikować surowego machine ID ani rozszerzać publicznych facts signera
o ten identyfikator.

Nowa ścieżka usuwa losowość `uuid.getnode()` również bez dostępnego MAC.
Nie rozwiązuje dostępu do prywatnej sesji D-Bus/Secret Service, responsywności
backendu, cross-role odmowy dostępu do kluczy ani brakującego niezależnego
private-control attestation. Nie otwiera fence, nie zmienia katalogu SQL,
semantyki BIND, polityki licencjonowania ani kontraktów Stage 10.

## Walidacja i bramki produkcyjne

Testy jednostkowe mają obejmować stabilność i walidację obu chronionych
składników, brak akceptacji losowego UUID, brak źródła/MAC, brak obniżenia
wymagań przy niedostępnym DMI, malformed input i odmowę przy zmienionym HWID
lub schemacie. Testy integracyjne na syntetycznych sekretach
sprawdzają szyfrowanie i odczyt w niezależnych procesach, zachowanie natywnego
read-only backendu oraz brak zapisu i generowania kluczy przy odmowie.
Regresje dotychczasowego fingerprintu, keyringu i licencji sprawdzają granicę
zgodności. Dokładne polecenia i wyniki testów oraz Ruff należą do opisu PR.

Środowisko przygotowania zmiany nie ma systemd jako PID 1. Próba utworzenia
jednorazowego user/network namespace przez `unshare` odmówiła zapisu do
`/proc/self/uid_map` na read-only filesystem. Testy z emulacją braku MAC lub
ograniczeniem socketów w procesie nie zastępują kwalifikacji rzeczywistego
`PrivateNetwork=yes`. Fizyczny test niezależnych restartów wewnątrz obu
zainstalowanych units, załadowanie root-sourced DMI credential do read-only
mountu, dokładny Secret Service i izolacja wszystkich ścieżek dostępu
pozostają bramkami produkcyjnymi. W tej pracy nie wykonano
produkcyjnych operacji ani Stage 10; Stage 9 pozostaje **NO-GO** niezależnie
od wyniku PR.
