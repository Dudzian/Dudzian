# Stage 9 — Trusted Installation & Identity Verification Gate

**READ-ONLY · BLOCKED · NOT_QUALIFIED · PRODUCTION_NO_GO**

Ta bramka porównuje instalację trzech usług `root_source`, `history_source`
i `cca` z jawnie dostarczonymi oczekiwaniami. Dostarcza zamknięty kontrakt,
kontrole plików i obserwację procesów. Nie instaluje usług, nie tworzy
principals ani kluczy i nie otwiera produkcyjnego SIGN/BIND, enrollmentu,
CCA publication lub Stage 10.

**Poprawny manifest i zgodność hosta z tym manifestem nie stanowią
niezależnej autoryzacji.** Oddzielny, chroniony pin pozwala wykrywać
rozbieżności z dostarczonymi oczekiwaniami, ale jego root ownership i
syntaktyczny identyfikator zatwierdzenia nie dowodzą procesu zatwierdzenia.
W repozytorium nie dostarczono kwalifikowanego produkcyjnego źródła
zaufania dla tej nowej instalacji. Dlatego także pełna zgodność zachowuje
`qualification=NOT_QUALIFIED`, `status=BLOCKED` i
`production_emission=BLOCKED`.

## Audyt i ponowne użycie

Przejrzano rzeczywiste źródła i odpowiednie kontrakty S9-RPI-01, 02, 06,
07, 21 oraz 26–38. Codebase Memory MCP nie był dostępny w katalogu narzędzi;
użyto celowanego `rg` i bezpośrednich odczytów. Brak indeksu nie został
zinterpretowany jako dowód kompletności.

| Źródło | Wykorzystana właściwość / zachowana granica |
| --- | --- |
| [Source/CCA preflight](../../deployment/stage9_native_linux_source_cca_preflight.py), S9-RPI-21 | Rozdział trzech ról, wyłączne UID/GID, brak dodatkowych grup. Jego nazwy, jednostki i ścieżki pozostają kandydatami, bez automatycznego zatwierdzenia. |
| [Diagnostic handoff](../../deployment/stage9_native_linux_diagnostic_handoff.py), S9-RPI-32 | Raportowanie zamkniętych kodów, zachowanie blokad i brak ujawniania prywatnych danych. Podraport nie jest atestacją. |
| [Unit fragment observer](../../deployment/stage9_native_linux_unit_fragment_observation.py), S9-RPI-26 | Wspólne metadane obiektów, deskryptorowe odczyty i porównania przed/po. Nowa warstwa przyjmuje jawne oczekiwania i kontroluje ACL; nie zatwierdza obserwowanych digestów. |
| [Process executable observer](../../deployment/stage9_native_linux_process_executable_observation.py), S9-RPI-27 | Granica między hashem obiektu `/proc/PID/exe` a pochodzeniem kodu. Nowa kontrola wiąże obiekt z otwartym plikiem instalacji; sam zgodny hash nie wystarcza. |
| [Host capability triage](../../deployment/stage9_native_linux_host_capability_triage.py), S9-RPI-33 | Dostępność prymitywów hosta nie kwalifikuje instalacji ani tożsamości usługi. |
| [Production signer IPC](../../bot_core/production_signer_ipc.py), S9-RPI-02 | Właściwa granica `SO_PEERPIDFD`; zwykły `pidfd_open(MainPID)` jej nie zastępuje. Bramka nie uruchamia signer IPC ani jego serwera. |
| [Independent CCA verifier](../../bot_core/independent_cca_decision.py), S9-RPI-07 | Jawna policy nie kwalifikuje własnego pochodzenia. Instalacja, wykonanie, trwałość i publication zachowują oddzielne wymagania. |
| [Cgroup observer](../../deployment/stage9_native_linux_cgroup_membership_observation.py) i [process identity observer](../../deployment/stage9_native_linux_process_identity_observation.py) | Ponowne użycie ograniczonego `systemctl`, parserów procfs/starttime/status i cgroup v2, zamiast kopiowania istniejących obserwatorów. |

[S9-RPI-01](stage9_production_root_proof_lifecycle_isolation.md) zachowuje
odrębny lifecycle i committed-close boundary.
[S9-RPI-06](stage9_authenticated_custody_mutation_source.md) wymaga odrębnych
Source/CCA credentials i wyłącznych grup bez supplementary memberships.
S9-RPI-26–33 dostarczają obserwacji, a S9-RPI-34–38 zamkniętych deklaracji
TEST i planu denial checks; żaden z tych wyników nie jest źródłem
produkcyjnego zatwierdzenia manifestu. Operator-reported fizyczne dowody
[Stage41B](stage9_stage41b_physical_signer_isolation.md) dotyczą syntetycznych
signerów i nie kwalifikują produkcyjnych Source/CCA.

## Zamknięty manifest V1

[Kontrakt](../../deployment/stage9_trusted_installation_contract.py) przyjmuje
ograniczone oryginalne bytes JSON. Digest dotyczy tych dokładnych bytes,
łącznie z ich formatowaniem. Nieznane pola, duplicate JSON keys, nieobsługiwana
wersja, niewłaściwe typy, booleans w polach liczbowych, niekanoniczne ścieżki
i powtórzone identyfikatory powodują odmowę.

| Zakres | Dokładne pola |
| --- | --- |
| Dokument | `schema`, `version`, `installation_id`, `roles` |
| Rola | `role`, `installation_identity`, `unit`, `principal`, `uid`, `gid`, `supplementary_groups`, `unit_file`, `executable`, `control_group` |
| Plik `unit_file` / `executable` | `path`, `sha256`, `uid`, `gid`, `mode` |

Wymagane są `schema=CryptoHunter.SourceCCAInstallationV1`, integer
`version=1` i dokładnie trzy unikalne role. Każda rola ma własny
`installation_identity`, principal, jednostkę, UID, GID, plik wykonywalny
i cgroup. Jednostka musi mieć prostą nazwę `.service`, a nazwa pliku
jednostki odpowiadać tej nazwie. Cgroups nie mogą nakładać się jako rodzic
i potomek. UID/GID są niezerowe, rozłączne pomiędzy rolami;
`supplementary_groups` musi być dokładnie `[]` zgodnie z S9-RPI-06/21.

Pliki mają jawny lowercase SHA-256 i dokładny mode jako integer JSON.
W tym konserwatywnym profilu pliki jednostek i executables należą do
`0:0`, bez group/world write i bez bitów specjalnych. Executable musi
mieć owner execute, plik jednostki nie może mieć bitów execute. Kontrakt
nie dostarcza domyślnych principal names, ścieżek, digestów ani wartości
odczytanych z bieżącego hosta. Nie ma TOFU, generatora zaufanych pinów,
automatycznej naprawy ani opcji pomijania weryfikacji.

## Oddzielne oczekiwania i pochodzenie zaufania

Manifest i osobny plik pina muszą być dostarczone niezależnie od badanych
usług. Pin wiąże `manifest_sha256`, `installation_id` i `approval_id`.
Jego zamknięte pola to `schema`, `version`, `manifest_sha256`,
`installation_id`, `approval_id`, z dokładnym
`schema=CryptoHunter.SourceCCAInstallationApprovalPinV1` i integer `version=1`.
Ani manifest, ani pola pina nie zawierają prywatnych kluczy, credentials,
custody paths lub zgody na production GO.

Odczyt plików oczekiwań wymaga chronionej ancestry, root ownership,
braku niedozwolonych zapisów, symlinków, hardlinków i ACL. Różne nazwy
plików wskazujące ten sam obiekt nie są oddzielnymi źródłami. Badany
runtime principal nie może przepisać oczekiwań lub sam zatwierdzić
ich zawartości. Oba pliki muszą być też odrębne od plików badanej
instalacji. Kontrola odrzuca powtórzone `(st_dev, st_ino)` wśród
manifestu, pina i sześciu otwartych plików instalacji; obejmuje to
zaobserwowane aliasy bind-mount zachowujące pojedynczy hardlink.
Brak pina, niezgodny pin lub błąd odczytu oznacza odmowę.

Nawet zgodny chroniony pin uzyskuje wyłącznie
`PROTECTED_PIN_MATCHED_PROVENANCE_UNQUALIFIED`. `approval_id` jest
identyfikatorem korelacji, nie podpisem ani potwierdzoną zgodą operatora.
Nie istnieje argument CLI lub flaga JSON promująca ten stan do
`QUALIFIED` lub `PRODUCTION_READY`. Pomyślny test syntetyczny nie dostarcza
niezależnej historii zatwierdzenia.

## Odczytowy interfejs

Po niezależnym przygotowaniu i przeglądzie chronionych oczekiwań,
z wcześniej przejrzanego checkoutu:

```bash
python -m deployment.stage9_trusted_installation_identity_gate \
  --manifest /etc/cryptohunter/qualification/source-cca-installation.json \
  --approval-pin /etc/cryptohunter/qualification/source-cca-approval.json
```

To przykładowe lokalizacje, bez zatwierdzania istniejącej konfiguracji.
Polecenie tylko czyta i raportuje; nie tworzy tych plików, nie oblicza
zaufanych pinów z hosta i nie instaluje usług. Brak oczekiwań nie uruchamia
bootstrapu. CLI kończy się kodem **2** również przy pełnej zgodności,
ponieważ nie osiągnięto kwalifikacji produkcyjnej. Raportu ani exit code
nie należy używać jako production admission.

## Kontrole instalacji i procesu

[Warstwa plików](../../deployment/stage9_trusted_installation_files.py)
utrzymuje deskryptory katalogów i plików, odczytuje względem rodzica,
odmawia symlinków i wymaga pojedynczego hardlinku pliku. Kontroluje każdy
katalog ancestry, właścicieli, uprawnienia i ACL. Metadane przed otwarciem,
po otwarciu, przed/po odczycie i końcowe powiązanie parent/name muszą
opisywać ten sam obiekt. Odczyty mają limity rozmiaru; brak możliwości
sprawdzenia obiektu lub jego podmiana powodują odmowę bez częściowego PASS.
Trzysekundowy deadline pętli odczytu sprawdza upływ czasu pomiędzy
wywołaniami. Nie jest twardym timeoutem dowolnego blokującego syscallu
systemu plików. Fizyczne uruchomienie wymagające ścisłego limitu całej
operacji potrzebuje niezależnego zewnętrznego supervisora.

[Warstwa runtime](../../deployment/stage9_trusted_installation_runtime.py)
używa zamkniętego zestawu read-only właściwości systemd i ograniczonego
runnera. Wymaga właściwej aktywnej jednostki, principal i fragmentu,
braku drop-inów oraz zgodnego cgroup. Odczytuje rzeczywiste procfs UID/GID,
grupy i starttime oraz sprawdza obiekt wykonywalny względem otwartego
pliku instalacji. Powtórne kontrole wykrywają zaobserwowany restart,
exec/replacement lub zmianę deklaracji. Stale PID, nieobserwowalny proces,
niedostępny systemd i rozbieżności fail closed. Przydział nazwy konta
nie zastępuje kontroli rzeczywistych credentials procesu.
Bramka odrzuca też współdzielony PID oraz wspólny inode tego samego typu
namespace (`mnt`, `ipc`, `net`) pomiędzy badanymi rolami. Różne numery
namespace nie kwalifikują ich fizycznej izolacji ani efektywnego dostępu.

Kod nie kopiuje dowolnych backend records do raportu. Raport zawiera
zamknięte statusy i kody, bez unit bytes, executable bytes, exception text,
kluczy, credentials, environment lub prywatnych ścieżek. Sukces kontroli
roli oznacza zgodność sprawdzonych elementów z dostarczonymi oczekiwaniami;
nie tworzy produkcyjnej capability.

## Granice dowodu i wymagane fizyczne Ubuntu

Kontrole są sekwencyjne. Nie tworzą atomowego snapshotu całego hosta,
nie chronią przed przejętym kernelem/root/systemd, skoordynowanym rollbackiem
lub zmianami poza oknem kontroli. Rozdział UID/GID obejmuje tutaj trzy
badane role; sam nie dowodzi izolacji od wszystkich pozostałych authorities.
Zgodność `/proc/PID/exe` nie dowodzi integralności załadowanych bibliotek,
modułów interpretera, imported source, środowiska i całej runtime closure.
Samo cgroup/MainPID nie jest uwierzytelnionym kanałem IPC.

Pozostają jawnie `BLOCKED`:

1. Kwalifikowane niezależne źródło zatwierdzenia: chroniony provisioning,
   identyfikacja fizycznego hosta, provenance/review manifestu i pina,
   lifecycle zatwierdzenia i rollback trust floor.
2. Odczyt rzeczywistych, wcześniej zatwierdzonych usług na fizycznym Ubuntu:
   jednostki, overrides/drop-ins, ich bytes/digests, ancestry/ACL, credentials,
   cgroups, wykonane obiekty i zmiany procesu. Ten PR ich nie instaluje.
3. Uwierzytelniony Source/CCA socket peer z rzeczywistym `SO_PEERPIDFD`,
   a także pełna runtime closure i kwalifikacja mount/IPC/net namespaces.
4. Rzeczywiste denial checks i positive owner controls wobec innych Source,
   CCA, obu signerów, issuera oraz administratora lifecycle, łącznie
   z alternatywnymi ścieżkami dostępu, ACL, capabilities i native keyring.
5. Production private-control/source execution provenance, retained
   originals, credential eligibility/history, source durability i niezależny
   trwały log CCA z odtwarzaniem po awarii.
6. Rzeczywisty PostgreSQL 16 committed-close, katalog/uprawnienia,
   lock ordering i recovery oraz osobny przegląd formalnej integracji
   addendum transakcyjnego A+C.

Testy jednostkowe używają syntetycznych danych i kontrolowanych plików.
Sprawdzają kontrakt, kontrole bezpieczeństwa i odmowy; nie dowodzą
fizycznej separacji. Nie wymagają `sudo` ani uruchomienia jako root.
Wyniki testów i coverage należy raportować dla faktycznie wykonanego
commitu; historyczna macierz PostgreSQL z dwóch rewizji nie zastępuje
pełnej regresji jednej rewizji.

Canonical `deployment/stage9_current_status.json`, zamrożone kontrakty M0.5,
status integracji A+C i branch protections pozostają bez zmian.
Rozszerzenia CI są wyłącznie addytywne: istniejące Ruff/format,
konfigurowany mypy, testy Stage 9 i połączona branch coverage obejmują nowe
moduły i testy bez zmiany progów, timeoutów lub gate semantics.

Osobny [operator fizycznego testu Source/CCA](stage9_source_cca_physical_test_operator.md)
przygotowuje syntetyczny pakiet i umożliwia świadome uruchomienie trzech
tymczasowych usług wyłącznie przez operatora na fizycznym Dellu. Ponownie
wykorzystuje tę odczytową bramkę; nie dodaje do niej instalacji ani
produkcyjnego źródła zaufania. `MATCHED` nadal zachowuje kod CLI 2,
`NOT_QUALIFIED` i `production_emission=BLOCKED`.
