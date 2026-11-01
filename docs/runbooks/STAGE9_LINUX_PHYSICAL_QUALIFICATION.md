# Stage 9 — procedura dowodowa Dell, Ubuntu 24.04

Procedura przygotowuje i przekazuje dowody platformowe. Nie uruchomiono jej
fizycznie w ramach tego PR. Status pozostaje
`Stage 9 = IN_PROGRESS / PRODUCTION_NO_GO`; testy oprogramowania, odczytowa
diagnostyka i wyniki TEST ONLY nie przyznają produkcyjnego PASS.

Wykonawcą jest operator fizycznego Dell Latitude 5490 z Ubuntu 24.04 x86_64.
Nie trzeba instalować GitHub Actions runnera na laptopie. CI przyjmuje
ograniczony pakiet obserwacji, weryfikuje go niezależnie i osobno podejmuje
decyzję bramki. Model zaufania i audyt ograniczeń opisuje
[kontrakt dowodowy](../architecture/stage9_linux_physical_evidence.md).

## Zasady przerwania i zgody

Nie wykonywać całego dokumentu przez „uruchom wszystkie”. Każda faza jest
osobnym krokiem. Zgoda na diagnostykę nie oznacza zgody na instalację,
zatrzymanie usług, aktualizację, symulację awarii, restart ani usunięcie danych.
Administrator przed każdym takim działaniem musi świadomie zaakceptować
konkretny host, zakres zasobów, dokładne polecenie i sposób odzyskania.
Nie używać produkcyjnych kluczy, baz ani poświadczeń.

Przerwać przy obcej instalacji, kolizji kont/UID/GID, nieznanym obiekcie,
niezgodnym hashu, niepełnej obserwacji, niedostępnej sondzie, odmowie,
timeout, nieudanym cleanup lub wygaśnięciu wyzwania. Nie usuwać kolidujących
obiektów ani nie edytować raportu dla otrzymania PASS. Zachować nieudany raport
bez sekretów; nowa próba wymaga nowego wyzwania i izolowanego środowiska.

## FAZA A — diagnostyka tylko do odczytu

Uruchomić zwykłym kontem z czystego, uprzednio przejrzanego checkoutu.
`ZATWIERDZONY_COMMIT` oznacza pełny SHA kwalifikowanego commitu. Nie używać
ruchomej nazwy gałęzi jako tożsamości wykonania.

```sh
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all
uname -srmo
cat /etc/os-release
cat /proc/1/comm
systemd-detect-virt
systemctl --version
```

Oczekiwane są Ubuntu 24.04, x86_64, systemd jako PID 1, brak kontenera/VM oraz
zgodność HEAD z uprzednio zatwierdzonym SHA. `systemd-detect-virt` zwraca kod
1 dla braku wykrytej wirtualizacji; sam ten wynik nie jest atestacją fizyczności
i wymaga potwierdzenia operatora oraz zatwierdzenia hosta poza raportem.
Nie publikować pełnych danych identyfikacyjnych hosta w otwartym issue.

Istniejące sondy Stage 9 są odczytowe i celowo zwracają kod **2**:

```sh
python3 -B -m deployment.stage9_native_linux_host_capability_triage
python3 -B -m deployment.stage9_native_linux_source_cca_preflight
```

Wynik to diagnostyka `BLOCKED`/`NOT_QUALIFIED`, także przy dostępnych
możliwościach. Nie zmieniać kodu wyjścia na 0 ani nie przedstawiać wyników jako
LIVE. Diagnostyka Source/CCA ma własne blokady, odrębne od kwalifikacji platformy.

## FAZA B — przygotowanie izolowanego środowiska testowego

Przed jakąkolwiek zmianą administrator osobno zatwierdza dedykowany system
testowy na Dellu, kopię danych i plan powrotu. Ubuntu instalować na przeznaczonym
do testów nośniku/instalacji bez produkcyjnego custody. Nie przebudowywać
aktualnego systemu ani partycji bez odrębnej zgody. Zwykły katalog, chroot,
kontener i syntetyczne usługi Source/CCA nie zastępują fizycznego testu.

Przygotowanie musi zapewnić:

1. Chroniony checkout dokładnego, przejrzanego commitu; właściciela root,
   brak zapisu dla nieuprawnionych i bezpiecznych przodków ścieżki.
2. Uprzednio zaufany interpreter, biblioteki, `.pth`, `sitecustomize`, bytecode,
   środowisko importów oraz systemowe narzędzia. Kontrola kodu po jego imporcie
   nie ustanawia zaufania do startu. Samo `-B` nie blokuje odczytu `.pyc`.
   Przygotowane środowisko musi zawierać zatwierdzone `cryptography` i
   `psycopg` 3. Jeśli używa chronionego venv, wszystkie dalsze wywołania
   `/usr/bin/python3` zastąpić jednym zatwierdzonym interpreterem tego venv;
   jego hash musi odpowiadać polityce także po restarcie.
3. Odrębne konta usług i UID/GID, chronione ścieżki konfiguracji/stanu/IPC,
   oddzielny klaster PostgreSQL i dokładne peer/`pg_ident`. Brak jakiegokolwiek
   połączenia do produkcyjnej bazy.
4. Zatwierdzony artefakt instalacyjny, jego manifest, digest konfiguracji,
   plan aktualizacji oraz oddzielny kontrolowany przypadek jej niepowodzenia.
   Każdy dotyczy dokładnej kwalifikowanej rewizji, a wcześniejsza wersja użyta
   w aktualizacji jest jawnie zidentyfikowana.
5. Osobny, chroniony klucz dowodowy producenta i jego klucz publiczny
   zatwierdzony przez weryfikatora poza raportem. Nie używać klucza Root Proof,
   signera produkcyjnego ani klucza wygenerowanego doraźnie przez test jako
   rzekomego trust anchor.
6. Nowe, ograniczone czasowo wyzwanie CI związane z dokładną rewizją,
   artefaktem, konfiguracją, hostem i wykonaniem workflow. Nie ponawiać użytego
   wyzwania po awarii.

Konfiguracja z manifestu jest przygotowywana i chroniona w tej fazie:
producent wymaga jej rzeczywistych bajtów już przy baseline. Jednostki,
programy i katalog trwałości objęte `clean_paths` muszą pozostawać nieobecne
do czystej instalacji. Katalog trwałości jest wydzielony pod
`/var/lib/cryptohunter-stage9-qualification/<id>`; jego rodzice mają właściciela
root i brak zapisu dla grupy/innych. Katalog bundle i artefakty trzymać poza
czystym checkoutem.

Po niezależnym zweryfikowaniu chronionego kodu i środowiska administrator
może osobno zatwierdzić rozszerzony, nadal odczytowy odczyt tożsamości hosta:

```sh
cd /opt/ch-s9-operator
sudo /usr/bin/python3 -B -m deployment.linux_physical_collector --diagnose
```

Zawiera digesty identyfikatorów, hash interpretera i boot ID; nie zawiera
prywatnego klucza. Odczyt wymaga root ze względu na systemowe dane DMI.
Nie wykonywać go jako root z checkoutu należącego do zwykłego konta.
Wyjście jest materiałem do niezależnego zatwierdzenia hosta, bez prawa LIVE.

Administrator repozytorium przed pierwszą próbą konfiguruje chronione
environments `stage9-linux-challenge` i `stage9-linux-evidence`. Każde wymaga
niepustego `required_reviewers`, `prevent_self_review=true` oraz
`deployment_branch_policy.protected_branches=true`. W obu ustawia identyczne
zmienne `STAGE9_LINUX_EVIDENCE_POLICY_JSON` i
`STAGE9_LINUX_EVIDENCE_POLICY_SHA256`, a w pierwszym dodatkowo
`STAGE9_LINUX_EVIDENCE_HOST_ID`. SHA polityki dotyczy kanonicznego JSON.
Pusta domyślna lista `hosts` w repozytorium zawsze odmawia kwalifikacji.
Zatwierdzenie wpisu hosta i artefaktów nie następuje przez skopiowanie wartości
z otrzymanego bundle.

Po niezależnym zatwierdzeniu gotowego manifestu instalacji oraz trzech
rzeczywistych pakietów `.deb` operator uruchamia wyłącznie ścieżkę Linux:

```sh
gh workflow run platform-deployment.yml --repo Dudzian/Dudzian --ref main \
  -f run_stage10=false -f run_linux_physical=true
gh run list --repo Dudzian/Dudzian --workflow platform-deployment.yml --limit 5
```

Zapisać rzeczywisty run ID, numer próby i pełny SHA tego przebiegu.
Niezależny reviewer zatwierdza najpierw job `linux-physical-challenge`.
Po jego sukcesie pobrać tylko artefakt tego konkretnego przebiegu:

```sh
qualification_run=ZAPISANY_NUMERYCZNY_RUN_ID
qualification_attempt=ZAPISANY_NUMERYCZNY_NUMER_PROBY
gh run download "$qualification_run" --repo Dudzian/Dudzian \
  --name "stage9-linux-challenge-$qualification_run-$qualification_attempt" \
  --dir ../work/stage9-challenge
```

Przed użyciem wyzwania porównać SHA, hosta, materiały, hash polityki i termin
ważności z osobno zatwierdzonymi oczekiwaniami. Wyzwanie jest związane z
`github.sha` tego run; późniejsza zmiana `main` go nie aktualizuje. Nie używać
automatycznie SHA najnowszego `main` zamiast SHA wyzwania.

Administrator osobno przenosi zweryfikowane `linux-challenge.json`,
`installation.json` oraz wyłącznie dedykowany klucz dowodowy do chronionego
katalogu, np. `/var/lib/ch-stage9-evidence/biezaca-proba`. Klucz jest surowym
32-bajtowym kluczem prywatnym Ed25519, plikiem root bez uprawnień dla innych;
nie wpisywać jego treści do terminala ani workflow. `installation.json` jest
dokładnym, zatwierdzonym obiektem `installation` z polityki, nie nową ręcznie
dopasowaną konfiguracją. Katalog dowodów nie może kolidować z `clean_paths`
instalacji. Przygotowanie tych ścieżek i importów jako root jest osobnym
działaniem administratora.

Przejrzeć cały plan zmian przed wykonaniem jako root. Istniejący
`deployment/install-production-local.sh` usuwa drzewa kodu i zmienia
PostgreSQL 16/main; nie uruchamiać go na aktualnym Dellu ani w fazie A.
Producent dowodów nie nadaje temu skryptowi bezpieczeństwa rollbacku.
Pakiet `stage9_source_cca_test_package.py` pozostaje TEST ONLY i nie jest
artefaktem kwalifikacji platformy.

## FAZA C — fizyczna integracja systemd

Siedem rekordów ma stałą kolejność: `baseline`, `clean_install`,
`systemd_lifecycle`, `pre_reboot`, `post_reboot`, `update`, `failed_update`.
Dlatego faza C rozpoczyna się od baseline i przeglądu planu systemd;
wykonanie lifecycle następuje po czystej instalacji w fazie D.
Nie zmieniać kolejności rekordów dla dopasowania nazw faz dokumentu.

W dalszych poleceniach `/opt/ch-s9-operator` oznacza uprzednio niezależnie
zweryfikowany, chroniony checkout dokładnej kwalifikowanej rewizji.
`ZATWIERDZONY_CHALLENGE_ID` zastąpić pełnym identyfikatorem uprzednio
zweryfikowanego wyzwania. Każde wywołanie `sudo` wymaga osobnej, świadomej
decyzji operatora; nie wykonywać ich pętlą ani skryptem bez zatrzymań.

```sh
cd /opt/ch-s9-operator
qualification_directory=/var/lib/ch-stage9-evidence/biezaca-proba
qualification_challenge_id=ZATWIERDZONY_CHALLENGE_ID
```

Po zgodzie na uprzywilejowany odczyt i zapis chronionego raportu baseline:

```sh
sudo /usr/bin/python3 -B -m deployment.linux_physical_collector \
  --challenge "$qualification_directory/linux-challenge.json" \
  --manifest "$qualification_directory/installation.json" \
  --bundle "$qualification_directory/bundle.json" \
  --private-key "$qualification_directory/evidence-key.raw" \
  --phase baseline --approve "$qualification_challenge_id:baseline"
```

Po osobnej zgodzie na uruchamianie, zatrzymywanie, restart i kontrolowaną awarię
wyłącznie zatwierdzonych jednostek zebrać obserwacje systemd i procesów.
Wymagane są rzeczywiste jednostki instalacji testowej, ich fragmenty i drop-in,
UID/GID/grupy, executable, cgroup, stan aktywności i autostart. Zweryfikować
odrębność tożsamości i ochronę IPC; sama deklaracja `User=` w pliku jednostki
nie wystarcza.

Sekwencja obejmuje start, ograniczony czasowo graceful stop, ponowny start,
manual restart, kontrolowaną awarię, nadzorowany powrót i brak pozostawionych
procesów. Zachować chroniony stan pośredni producenta. Nie podpisywać ręcznie
wpisanego `PASS` ani ogólnego journalu. Nie uruchamiać diagnostycznych signerów
w trybie produkcyjnym; ich istniejące blokady pozostają w mocy.

## FAZA D — czysta instalacja

Po odrębnej zgodzie na instalację i ewentualne tworzenie kont potwierdzić
brak poprzedniej instalacji w dedykowanym systemie, zainstalować dokładnie
zatwierdzony artefakt i zebrać wynik instalatora oraz stan po instalacji.
Sprawdzić digest zainstalowanych plików, konfigurację, ochronę katalogów,
powiązanie jednostek z artefaktem, rzeczywiste tożsamości i PostgreSQL/IPC.
Zapis „instalacja wykonana” bez obserwacji stanu przed i po nie wystarcza.

Nie usuwać danych, kont i jednostek, aby sztucznie uzyskać warunek czystej
instalacji. Nowy nośnik lub przywrócenie zatwierdzonego obrazu jest osobnym,
potencjalnie destrukcyjnym działaniem administratora.

Po zgodzie na instalację pakietu z przypiętego manifestu:

```sh
sudo /usr/bin/python3 -B -m deployment.linux_physical_collector \
  --challenge "$qualification_directory/linux-challenge.json" \
  --manifest "$qualification_directory/installation.json" \
  --bundle "$qualification_directory/bundle.json" \
  --private-key "$qualification_directory/evidence-key.raw" \
  --phase clean_install --approve "$qualification_challenge_id:clean_install"
```

Po sukcesie instalacji i kolejnej zgodzie na kontrolowane stop/start/restart
oraz awarię jednostek z manifestu wykonać część wykonawczą fazy C:

```sh
sudo /usr/bin/python3 -B -m deployment.linux_physical_collector \
  --challenge "$qualification_directory/linux-challenge.json" \
  --manifest "$qualification_directory/installation.json" \
  --bundle "$qualification_directory/bundle.json" \
  --private-key "$qualification_directory/evidence-key.raw" \
  --phase systemd_lifecycle --approve "$qualification_challenge_id:systemd_lifecycle"
```

## FAZA E — restart i lifecycle

Przed restartem producent zapisuje podpisany stan ciągłości: identyfikator
wyzwania i wykonania, tożsamość hosta, bieżący boot ID, sekwencję zdarzeń,
rewizję, artefakty, konfigurację i digest kontrolowanego trwałego stanu.
Wylogowanie, restart usługi ani zmiana pola boot ID w JSON nie zastępują
restartu systemu.

Po zgodzie na zapis stanu ciągłości wykonać:

```sh
sudo /usr/bin/python3 -B -m deployment.linux_physical_collector \
  --challenge "$qualification_directory/linux-challenge.json" \
  --manifest "$qualification_directory/installation.json" \
  --bundle "$qualification_directory/bundle.json" \
  --private-key "$qualification_directory/evidence-key.raw" \
  --phase pre_reboot --approve "$qualification_challenge_id:pre_reboot"
```

Po świadomej, odrębnej zgodzie operatora na restart wykonać go ręcznie:

```sh
sudo /usr/bin/systemctl reboot
```

Po ponownym zalogowaniu kontynuować to samo chronione wykonanie. Producent
musi potwierdzić ten sam host, nowy boot ID, autostart, zgodną tożsamość usług,
niezmieniony trwały stan i kontrolowany powrót. Utrata stanu pośredniego,
powtórzony boot ID lub inny host blokują dowód.

Po restarcie ponownie przejść do chronionego checkoutu, ustawić te same
`qualification_directory` i `qualification_challenge_id` z zachowanych,
zweryfikowanych danych. Po odrębnej zgodzie na obserwację:

```sh
sudo /usr/bin/python3 -B -m deployment.linux_physical_collector \
  --challenge "$qualification_directory/linux-challenge.json" \
  --manifest "$qualification_directory/installation.json" \
  --bundle "$qualification_directory/bundle.json" \
  --private-key "$qualification_directory/evidence-key.raw" \
  --phase post_reboot --approve "$qualification_challenge_id:post_reboot"
```

Aktualizacja wymaga osobnej zgody na dokładny, przejrzany plan. Zaobserwować
przejście z jawnie zatwierdzonej wcześniejszej wersji do kwalifikowanego
artefaktu, restart usług i zachowanie trwałego stanu. Następnie po kolejnej
zgodzie wykonać kontrolowaną nieudaną aktualizację i potwierdzić odmowę,
niezmieniony ostatni poprawny artefakt, konfigurację i stan oraz aktywne,
prawidłowo izolowane usługi. Nie symulować tego przez sam `FAIL`/`PASS` w JSON.
Brak zatwierdzonego mechanizmu bezpiecznej aktualizacji pozostawia
`LINUX_REBOOT_LIFECYCLE_UPDATE` zablokowane.

Po osobnej zgodzie na aktualizację:

```sh
sudo /usr/bin/python3 -B -m deployment.linux_physical_collector \
  --challenge "$qualification_directory/linux-challenge.json" \
  --manifest "$qualification_directory/installation.json" \
  --bundle "$qualification_directory/bundle.json" \
  --private-key "$qualification_directory/evidence-key.raw" \
  --phase update --approve "$qualification_challenge_id:update"
```

Po przejrzeniu oczekiwanego błędu trzeciego pakietu i kolejnej zgodzie
na kontrolowaną nieudaną aktualizację:

```sh
sudo /usr/bin/python3 -B -m deployment.linux_physical_collector \
  --challenge "$qualification_directory/linux-challenge.json" \
  --manifest "$qualification_directory/installation.json" \
  --bundle "$qualification_directory/bundle.json" \
  --private-key "$qualification_directory/evidence-key.raw" \
  --phase failed_update --approve "$qualification_challenge_id:failed_update"
```

Producent nie restartuje systemu samodzielnie. Token `--approve` wiąże zgodę
z konkretnym challenge i jedną fazą; nie zastępuje przeczytania planu działań.
Awaria może pozostawić zmieniony stan testowego hosta, dlatego nie ponawiać
fazy w ciemno i nie usuwać istniejącego bundle, by ukryć poprzednią próbę.

## FAZA F — niezależna weryfikacja dowodów

Przenieść zamknięty pakiet na maszynę weryfikatora. Klucz publiczny, polityka
zatwierdzonych producentów/hostów, wyzwanie i oczekiwane hashe muszą pochodzić
z niezależnego, chronionego źródła; nie z przyjętego pakietu.
Weryfikator sprawdza podpis, format i pełny zestaw etapów, kolejność, świeżość,
jednorazowość, zgodność rewizji, artefaktów, konfiguracji i hosta, restart oraz
ciągłość stanu. Decyzję odtwarza z obserwacji. Nie wykonuje dostarczonego kodu
i nie ufa wynikowi `PASS` zadeklarowanemu przez operatora.

Po odmowie zachować przyczynę. Nie przepodpisywać poprawionego ręcznie pliku.
TEST ONLY, brak fizycznego etapu i syntetyczna obserwacja są odmową LIVE.

Na odrębnej maszynie, z zaufanego checkoutu i niezależnie uzyskanej publicznej
polityki, można wykonać wstępną walidację. `qualification_sha` i
`qualification_policy_digest` ustawić na uprzednio zatwierdzone wartości.
Plik wynikowy musi być nowy:

```sh
python3 -B -m deployment.linux_physical_evidence verify \
  --bundle bundle.json --challenge linux-challenge.json \
  --policy approved-policy.json \
  --expected-policy-sha256 "$qualification_policy_digest" \
  --source-revision "$qualification_sha" \
  --ci-run-id "$qualification_run" --ci-run-attempt "$qualification_attempt" \
  --output linux-evidence-preview.json
```

To sprawdzenie nie zużywa challenge i nie jest decyzją bramki. Plik preview
nie jest dopuszczalnym zamiennikiem raw bundle. Na CI przesłać wyłącznie
zweryfikowany `bundle.json`, bez klucza, pełnych logów i sekretów. Przygotować
ograniczony transport gzip+Base64; podpisy nadal dotyczą surowych obserwacji:

```sh
python3 -B -m deployment.linux_evidence_handoff pack \
  --bundle bundle.json --output transport.txt
gh workflow run linux-evidence-ingress.yml --repo Dudzian/Dudzian --ref main \
  -f target_run_id="$qualification_run" \
  -f target_run_attempt="$qualification_attempt" \
  -F bundle=@transport.txt
```

Limit transportu wynosi **48 KiB**, a rozpakowanego dokumentu **256 KiB**.
Ingress odrzuca nadmierną dekompresję, błędną sumę kontrolną, dodatkowe
strumienie gzip i dane po końcu strumienia. Zaczekać na sukces osobnego ingress, sprawdzić jego
run i nie przesyłać drugiej kopii pod tę samą nazwą; więcej niż jeden artefakt
jest odmową. Dopiero potem niezależny reviewer zatwierdza oczekujący
`linux-physical-validation` w pierwotnym przebiegu. Job ponownie pobiera raw
bundle, politykę i challenge oraz weryfikuje je samodzielnie.

## FAZA G — końcowa kwalifikacja

Niezależny reviewer osobno zatwierdza job `linux-production-gate` w environment
`stage9-linux-evidence`. Job wymaga sukcesu walidatora i wszystkich trzech
systemów macierzy core. Ponownie weryfikuje surowe dowody przez
`platform_readiness`, przekazując `--linux-bundle`, `--linux-challenge`,
`--linux-policy`, `--linux-policy-sha256`, `--expected-ci-run-attempt`
i chroniony `--linux-replay-ledger`. Rejestru nie usuwać dla wymuszenia retry.
Próba ponownego użycia, inny run/attempt lub wygasły challenge oznaczają odmowę.
Rejestr w `RUNNER_TEMP` zapewnia atomowe zużycie w danym zadaniu, a wyzwanie
powiązane z nonce, run ID i numerem próby blokuje przeniesienie dowodów do
nowego przebiegu lub ponowionego zadania. Rejestr runnera nie jest trwałym
archiwum audytowym po usunięciu maszyny. Lokalna bramka poza tym workflow
wymaga trwałego, chronionego rejestru; nie wolno tworzyć nowego pustego
rejestru dla ponownego użycia wcześniejszego wyzwania.

Decyzja bramki następuje dopiero po niezależnej weryfikacji i uzyskaniu
aktualnych dowodów pełnej macierzy core dla tej samej rewizji. Brak dowodu,
odmowa walidatora lub nieudany job blokują `LINUX_PRODUCTION_READY`.
Nie usuwać wymagań z `platform_readiness.json` i nie przepisywać historycznych
statusów. Kwalifikacja platformy nie przyznaje uprawnień do produkcyjnego
Source/CCA, custody, SIGN/BIND, enrollment ani Stage 10.

Osobna inspekcja cleanup jest wymagana przed przywróceniem testowej maszyny
do innego użycia. Usuwanie zasobów albo odtwarzanie obrazu wymaga kolejnej
świadomej zgody administratora. Nie usuwać kluczy, katalogów ani kont
o niepotwierdzonym pochodzeniu. Wynik kwalifikacji nie oznacza zgody na merge
PR ani zmiany Stage 9 na COMPLETE.

## Decyzja przed pierwszym wykonaniem

**Fizyczna kwalifikacja produkcyjna: NO-GO**, dopóki administratorzy nie
zatwierdzą hosta i klucza dowodowego, chronionych environments, pełnego
manifestu instalacji oraz trzech rzeczywistych artefaktów z bezpiecznym
planem aktualizacji. Obecna domyślna polityka nie zawiera zatwierdzonego hosta.
Diagnostyczne signery, custody, Source/CCA oraz istniejące fixture TEST ONLY
nie spełniają tych warunków. Można rozpocząć fazę A i przygotowanie
infrastruktury; polecenia uprzywilejowane w dalszych fazach są instrukcją
przyszłego wykonania po spełnieniu warunków i osobnych zgodach operatora.
