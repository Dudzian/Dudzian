# Stage 9 — zaufana ścieżka dowodowa Linux

Ten dokument dotyczy dowodów platformowych dla trzech wymagań:
`LINUX_SYSTEMD_INTEGRATION`, `LINUX_CLEAN_INSTALL` oraz
`LINUX_REBOOT_LIFECYCLE_UPDATE`. Implementacja i testy infrastruktury nie są
fizyczną kwalifikacją. `Stage 9 = IN_PROGRESS / PRODUCTION_NO_GO` pozostaje
bez zmian; nie powstaje uprawnienie do SIGN/BIND, enrollment, publikacji CCA
ani wykonania Stage 10.

## Stan po remediacji PR #3233

Podpisane metadane dowolnego pakietu i dwóch usług nie identyfikują produktu.
Pełna kwalifikacja wymaga teraz `installation.release_identity`: certyfikatu
wydania podpisanego przez niezależny klucz z `policy.release_authorities`.
Klucz autorytetu wydania musi różnić się od wszystkich kluczy producentów
hosta. Publiczne identyfikatory produktu, profilu oraz wymaganych kontraktów
określa `linux_release_identity.py`; nazwy pakietów, jednostek i programów
pochodzą z zatwierdzonego wydania. Certyfikat wiąże rewizję, trzy artefakty,
wersje, programy komponentów i digest profilu instalacji bez samego certyfikatu.
Jest potwierdzeniem tożsamości wydania, a nie wykonania testów funkcjonalnych.

Brakuje bezpiecznych adapterów rzeczywistych usług potwierdzających pełne
uwierzytelnianie IPC i uprawnienia PostgreSQL. Dlatego `verify_bundle()`
nie może obecnie zwrócić produkcyjnych wyników PASS, także przy poprawnym
certyfikacie i kompletnym podpisanym łańcuchu. Nie ma przełącznika ani flagi
JSON pozwalającej usunąć tę blokadę. Natywna ścieżka kolektora odmawia faz
zmieniających host przed wywołaniem instalatora lub sterowania usługami.
Instalator `install-production-local.sh` pozostaje odrębnym zakresem naprawy.

`linux_physical_ipc.py` odczytuje rzeczywiste `SO_PEERCRED` już zestawionego
połączenia i przypina partnera przez socket-bound `SO_PEERPIDFD`, PID,
UID/GID, czas startu i program. Same te fakty nie dowodzą decyzji serwera
aplikacji ani odmowy wobec obcej tożsamości. Istniejący signer ma wyłącznie
diagnostyczny protokół zwracający BLOCKED; nie wysyłamy do usług dowolnych
ramek i nie zastępujemy produktu serwerem echo. Brak możliwości pidfd
oznacza odmowę bez fallbacku do samego PID.

Obserwacja PostgreSQL administratora jest jawnie oddzielona jako
`postgres.administrative_observation`. `postgres.service_qualification`
pozostaje BLOCKED. Odczytowa sonda `linux_physical_postgresql.py` sprawdza
tożsamość własnego nieadministracyjnego procesu, peer, role sesji, backend PID
i faktyczną odmowę odczytu chronionego katalogu. Osobna sonda uruchomiona
z UID usługi nadal nie jest dowodem połączenia tej konkretnej usługi;
pełna macierz operacji produktu i związanie połączeń runtime/verifier
wymagają odrębnego zatwierdzonego adaptera.

Polityka jest wersjonowana na osobnej chronionej gałęzi
`stage9-linux-evidence-policy`, w pliku `policy.json`. To usuwa cykliczne
wymaganie zapisania SHA commitu kodu we własnej zawartości. Workflow pobiera
dokładny, zatwierdzony commit polityki i sprawdza jego digest oraz aktualność.
Kontrola GitHub potwierdza rzeczywistą ochronę obu gałęzi, CODEOWNERS,
wymagane review i CI oraz niezależne zatwierdzanie environments.
Właściciel wdraża ustawienia według [osobnej procedury](../runbooks/STAGE9_GITHUB_EVIDENCE_CONTROLS.md).
Ta remediacja nie zmienia ustawień repozytorium.

## Audyt kontraktu sprzed tej zmiany

Wnioski poniżej pochodzą z bezpośredniego odczytu źródeł. Codebase Memory MCP
i jego CLI były niedostępne w środowisku tej sesji; nie wykonano zapytania
grafowego ani kontroli pokrycia indeksu. Zakres audytu obejmuje wskazane
kontrakty i powiązane implementacje, a nie twierdzenie o kompletności całego
repozytorium.

| Źródło | Zaobserwowana luka lub zachowana granica |
| --- | --- |
| `deployment/platform_evidence.py` | Istnieją producenci macierzy core i dowodów Windows. Brakuje producenta trzech fizycznych wymagań Linux. Przekazane SHA i identyfikator CI nie są samodzielnym dowodem wykonania. |
| `deployment/platform_readiness.py` | Zamrożona lista sześciu wymagań Linux jest poprawna. Dotychczasowa walidacja envelope porównuje deklarowane SHA, provider, run i OS oraz składnię czasu; nie uwierzytelnia hosta ani producenta, nie sprawdza świeżości, challenge i wcześniejszego użycia, nie odtwarza wyniku z obserwacji. |
| `deployment/platform_evidence.schema.json` | Envelope V1 opisuje deklaracje wyników, ale nie ustanawia pochodzenia. Sama zgodność ze schematem nie może awansować ręcznego JSON do LIVE. |
| `deployment/platform_readiness.json` | Trzy wymagania fizyczne pozostają niezweryfikowane. Dwa kontrakty statyczne mają inny zakres. Historyczne wpisy nie są dowodami dla nowej rewizji i nie należy ich przepisywać. |
| `.github/workflows/platform-deployment.yml` | Dotychczasowy Linux gate otrzymuje tylko macierz core; brak bezpiecznego przekazania fizycznego wykonania i niezależnej weryfikacji. Laptop operatora nie musi być GitHub runnerem. |
| `deployment/platforms/linux.py` oraz `bot_core/freshness_deployment_qualification.py` | Istniejąca kwalifikacja POSIX/systemd/PostgreSQL jest warstwą sprawdzania instalacji, a nie podpisanym dowodem całego clean-install/reboot/update. Jej wymagania pozostają obowiązkowe. |
| `deployment/systemd/cryptohunter-freshness-verifier.service` | Jednostka określa użytkownika, grupę IPC, ochronę systemu i restart. Odczyt pliku nie dowodzi skutecznej konfiguracji uruchomionej usługi, rzeczywistego UID/GID ani peer authentication PostgreSQL. |
| `deployment/install-production-local.sh` | Skrypt wymaga root, tworzy rzeczywiste konta, usuwa istniejące drzewa kodu i zmienia konfigurację PostgreSQL 16/main. Nie wytwarza receipt, nie zapewnia rollbacku ani kompletnego testu uruchomionej usługi. Nie używać go automatycznie na zwykłej instalacji Della. |
| `deploy/packaging/desktop_installer.py` | Budowa ZIP i syntetyczny smoke z `KBOT_FAKE_FINGERPRINT` są kontraktem pakietu, bez dowodu fizycznej instalacji. |
| `core/update/installer.py`, `scripts/offline_update.py`, `scripts/desktop_updater.py` | Aktualizacje modeli/strategii i archiwum ZIP nie ustanawiają uprzywilejowanego handoff, atomowej aktualizacji usług ani bezpiecznego rollbacku backendu. Scenariusz fizyczny wymaga odrębnie zatwierdzonego planu sukcesu i kontrolowanej awarii. |
| Jednostki `deployment/systemd/cryptohunter-*-signer.service` | Są diagnostyczne, blokują produkcyjne podpisywanie i nie są dowodem gotowego autostartu/lifecycle produkcji. |
| `deployment/stage9_trusted_installation_runtime.py`, obserwatory `stage9_native_linux_*` | Rzeczywiste sondy systemd, NSS, procfs, executable, cgroup, pidfd i namespace są użyteczne. Niedostępność lub zmiana obserwowanego obiektu wymaga odmowy. Snapshot sam nie uwierzytelnia producenta ani całego hosta. |
| `deployment/stage9_trusted_installation_identity_gate.py` | Także `installation_checks=MATCHED` pozostawia `NOT_QUALIFIED` i trwałe blokady zaufania do hosta, socket-bound peer, skutecznej izolacji, kodu, custody i PostgreSQL/CCA. Nowa ścieżka platformowa nie usuwa tych blokad. |
| `deployment/stage9_source_cca_test_package.py`, `scripts/qualify_stage9_source_cca_installation.py` | Pakiet i operator są jawnie TEST ONLY/SYNTHETIC. Kontrolowane procesy nie implementują produkcyjnych Source/CCA. Ich wyniki nie spełniają trzech pozycji LIVE Linux. |
| `deployment/stage9_native_linux_source_cca_preflight.py` | Odczytowy preflight zachowuje `UNTRUSTED_LIVE_OBSERVATION`, `BLOCKED` i brak kwalifikacji. Brak wykrytej kolizji nie jest zatwierdzeniem hosta. |
| `tests/deployment/test_platform_readiness.py`, `tests/deployment/test_platform_evidence.py` | Syntetyczne fixture sprawdzają decyzje o deklarowanych envelope; ich pozytywny wynik jest regresją oprogramowania, bez kwalifikacji fizycznej. Potrzebne są odmowy podpisu, pochodzenia, freshness/replay, hosta, restartu, rewizji i artefaktów. |

Wszystkie sześć wymagań `LINUX_PRODUCTION_READY` pozostają obowiązkowe:
macierz core, kwalifikacja statyczna, integracja systemd, statyczny kontrakt
instalatora, czysta instalacja i reboot/lifecycle/update. Zaufany raport
fizyczny nie zastępuje aktualnej macierzy core ani kontraktów statycznych.

## Granica dowodu fizycznego

Pochodzenie musi wynikać z zatwierdzenia poza samym raportem: przypięty
producent i jego klucz publiczny, zaakceptowany host i środowisko, dokładna
rewizja oraz jednorazowe wyzwanie weryfikatora. Pola `source_revision`,
`ci_run_id` i `runner_os` są sprawdzanymi deklaracjami, nigdy źródłem zaufania.
Pakiet odebrany od operatora jest nieufnym wejściem, także gdy jego transport
zakończył się powodzeniem.

Podpis poświadcza odpowiedzialność zatwierdzonego producenta za obserwacje.
Nie dowodzi fizyczności sprzętu ani uczciwości skompromitowanego administratora.
Zatwierdzenie producenta wymaga osobnego przeglądu przygotowania Dell, ochrony
klucza, checkoutu, interpretera, importów, narzędzi systemowych oraz konfiguracji
testowej. Weryfikator na innej maszynie nie uruchamia pakietu, skryptów,
instalatora ani poleceń dostarczonych przez raport.

Kod podpisujący nie może udostępniać operacji „podpisz dowolny JSON”. Musi
zbierać obserwacje zamkniętego scenariusza. Sam podpis pod ręcznym `PASS`,
podstawione logi lub syntetyczny backend nie ustanawiają wyniku LIVE.
Klucz producenta jest odrębnym kluczem dowodowym; rzeczywiste klucze
produkcyjnego custody nie biorą udziału w tej procedurze.

## Zaimplementowany protokół

`deployment/linux_physical_evidence.py` definiuje zamknięty protokół.
`deployment/linux_physical_observations.py` niezależnie interpretuje jego
obserwacje. Polityka `deployment/linux_physical_trust_policy.json` ma domyślnie
pustą listę `hosts`: repozytorium nie zatwierdza Della, klucza ani fixture.
Nie można rozpocząć kwalifikacji LIVE z tą domyślną polityką.

Osobno zatwierdzona polityka ma cel `LIVE_PHYSICAL_LINUX`, wersję 1,
`max_age_seconds` od 1 do 86400 i najwyżej 32 hosty. Każdy wpis przypina:

- `host_id`, SHA-256 machine-id i DMI product UUID, SHA-256 interpretera
  oraz odrębny publiczny klucz Ed25519; identyfikatory i klucze nie mogą
  powtarzać się między hostami;
- pełny, zamknięty kontrakt `installation`: ścieżkę kodu, trzy pakiety `.deb`,
  odrębne tożsamości runtime/verifier/administrator, jednostki i programy,
  chronioną konfigurację, IPC, katalog testowej trwałości, warunki czystej
  instalacji oraz izolowaną konfigurację PostgreSQL;
- `materials`: pełny 40-znakowy `source_revision` i 64-znakowe SHA-256
  kolektora, artefaktu bazowego, aktualizacji, nieudanej aktualizacji,
  konfiguracji i kontraktu instalacji. Trzy artefakty muszą być różne.

Oczekiwany SHA-256 polityki jest skrótem **kanonicznego obiektu JSON**,
wyliczanym przez `digest(read_json(path))`, a nie `sha256sum` pliku z dowolnym
formatowaniem. Polityka i ten skrót podlegają osobnemu zatwierdzeniu poza
pakietem dowodowym. Hash kontraktu instalacji także wiąże jego kanoniczny
obiekt. Nie wolno dopasować trust anchor do otrzymanego raportu.

Wyzwanie CI przypina repozytorium `Dudzian/Dudzian`, dokładną rewizję,
`ci_run_id`, `ci_run_attempt`, hosta, politykę i materiały. Zawiera losowy
256-bitowy nonce, czas wystawienia, termin ważności oraz `challenge_id`,
będący skrótem całego wyzwania bez samego tego pola. Termin nie przekracza
24 godzin ani limitu polityki. Walidator wymaga aktualnego, niewygasłego
wyzwania. Czas sprawdza według własnego zegara; zegar producenta nie jest
źródłem autorytetu.

Pakiet zawiera dokładnie siedem rekordów w kolejności:

```text
baseline -> clean_install -> systemd_lifecycle -> pre_reboot
         -> post_reboot -> update -> failed_update
```

Każdy rekord ma numer sekwencji, fazę, challenge, czas, hosta, materiały,
surowe obserwacje i podpis Ed25519 nad kanonicznym payload. Pierwszy wskazuje
zerowy poprzedni skrót, kolejne SHA-256 całego poprzedniego podpisanego rekordu.
Wymagane są monotoniczne czasy w oknie wyzwania, zgodna tożsamość i interpreter.
Pierwsze cztery rekordy mają ten sam boot ID; ostatnie trzy drugi, różny boot ID.
Tożsamość hosta i interpretera zachowuje ciągłość przez restart. Sama etykieta
`LIVE`, podpis lub syntaktycznie poprawny boot ID nie wystarczają bez pełnego
sprawdzenia obserwacji.

Docelowy envelope jest prezentacją wyniku, bez samodzielnego prawa kwalifikacji.
Obecny weryfikator odmawia jego emisji z powodu opisanych blokad produktu,
IPC i PostgreSQL. Sam poprawny podpis transportu może zostać zweryfikowany
oddzielnie, ale nie nadaje kwalifikacji.
`platform_readiness` ponownie weryfikuje surowy bundle, wyzwanie i przypiętą
politykę; ręcznie wgrany wynik walidatora nie otwiera bramki.
`consume_challenge()` atomowo rezerwuje challenge w SQLite z unikalnym kluczem.
Rejestr musi być chroniony i zachowany dla domeny decyzji; jego brak,
niedostępność lub ponowne użycie powodują odmowę. Nie usuwać rejestru dla retry.
Nowa próba CI otrzymuje nowe wyzwanie i `ci_run_attempt`.

## Przekazanie do CI

Fizyczne wykonanie odbywa się poza runnerem. Manualny przebieg
`platform-deployment.yml` wystawia wyzwanie po zatwierdzeniu chronionego
environment `stage9-linux-challenge`. Osobny manualny
`linux-evidence-ingress.yml` zapisuje literalne, nieufne bajty bundle
i publikuje artefakt transportowy. Limit transportu gzip+Base64 wynosi
48 KiB, a rozpakowanego bundle 256 KiB; transport
odrzuca duplikaty pól i niedozwolone stałe JSON. Nie używa `eval`, powłoki
ani kodu z otrzymanego raportu.

Po osobnym zatwierdzeniu environment `stage9-linux-evidence` niezależny job
pobiera dokładnie jeden artefakt przypisany do bieżącego run/attempt.
Kontroluje jego pochodzenie z manualnego workflow ingress na `main` tego
samego repozytorium, kompletność katalogu, ważność i rozmiar. ZIP może
zawierać tylko `bundle.json`; katalogi, symlinki i ścieżki wyjścia są odmową.
Podpisane adresy magazynu są tylko transportem i nie otrzymują tokena GitHub.
Artefakt GitHub nie jest trust anchor — po pobraniu walidacja kryptograficzna
i semantyczna pozostaje obowiązkowa.

Chronione environments wymagają niezależnych reviewerów, zakazu self-review,
wyłączonego administrator bypass i rzeczywistej ochrony gałęzi.
Sama flaga protected_branches nie wystarcza, gdy main jest niechroniony.
Brak wymaganej konfiguracji nie może skutkować
przejściem na słabszą ścieżkę. Konfiguracja polityki, hosta i sekretów
środowiska jest zadaniem administratora repozytorium, poza tym PR.
Produkcję rozstrzyga osobny job bramki z aktualną macierzą core i ponowną
walidacją raw bundle; upload i walidacja nie są samodzielną decyzją bramki.

## Niezaspokojone wymagania przed LIVE

W repozytorium nie ma zatwierdzonego wpisu Della, dedykowanego klucza dowodowego,
produkcyjnego zestawu trzech pakietów `.deb` ani kontraktu ich instalacji
spełniającego nowe wymagania. Istniejących pakietów TEST ONLY nie wolno
wpisać do polityki jako instalacji produkcyjnej. Trzeba niezależnie przejrzeć
i zatwierdzić instalator, jednostki, działający backend, plan aktualizacji,
przypadek awarii, zainstalowane pliki, konfigurację, uprawnienia i izolację.
Szczególnie diagnostyczne signery i custody zachowują wcześniejsze blokady.
Do spełnienia tych warunków dopuszczalna jest diagnostyka i przygotowanie
infrastruktury, ale fizyczna kwalifikacja produkcyjna pozostaje NO-GO.

## Zachowanie przy niepewności

Niepełne dane, nieznane pola, nieudany etap, błędna kolejność, zmiana hosta
lub tożsamości po restarcie, brak nowego boot ID, brak zgodności kodu,
konfiguracji lub artefaktu, TEST ONLY, obcy klucz, nieważny podpis oraz
wygasłe albo użyte wyzwanie prowadzą do odmowy. Nie wolno poprawiać raportu,
usuwać nieudanych pozycji ani po awarii odtwarzać pozytywnego wyniku z opisu
operatora. Nowa próba wymaga nowego wyzwania.

Raporty zawierają wyłącznie zamknięty zbiór dozwolonych obserwacji i digesty.
Nie zbierać pełnego środowiska procesu, parametrów DSN z hasłem, plików
prywatnych kluczy, pełnego journalu ani konfiguracji mogącej zawierać sekrety.
Hash niejawnego pliku nie oznacza zgody na umieszczenie jego treści w raporcie.

Procedura operatorska znajduje się w
[runbooku Dell Ubuntu 24.04](../runbooks/STAGE9_LINUX_PHYSICAL_QUALIFICATION.md).
