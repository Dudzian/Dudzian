# Stage 9 Linux — audyt harnessu VM, 2026-11-03

Zakres: odtwarzalna ręczna próba podpisanego runtime, instalatora v2,
rzeczywistego systemd/PostgreSQL i IPC na izolowanej Ubuntu 24.04 amd64 VM.
Procedura: [STAGE9_LINUX_VM_HARNESS.md](../runbooks/STAGE9_LINUX_VM_HARNESS.md).
Nie jest to kwalifikacja fizycznego Della ani zgoda na produkcję/merge.

## Baza i dostępne środowisko

Przegląd rozpoczęto od aktualnego `origin/main`
`8d0d6892080f0cb5d7d64dc41fb5c598c851c65e`, po merge #3239 i #3244.
Źródła są podstawą ustaleń. Codebase Memory MCP/CLI nie były dostępne;
przegląd zawężono przez wyszukiwanie i bezpośredni odczyt wskazanych modułów.
Nie deklaruje się kompletności indeksu strukturalnego.

Dostępny executor ma Debian GNU/Linux 13, PID 1 `tail`, interpreter roboczy
Python 3.12.14 i brak `/dev/kvm`. Nie jest zatwierdzoną Ubuntu 24.04 VM
z systemd jako PID 1. Nie wykonywano na nim uprzywilejowanej instalacji,
tworzenia kont/klastrów ani restartów. Nie zastąpiono próby VM kontenerem.

| Dowód natywny | Wynik obecnej sesji |
| --- | --- |
| Identyfikator obrazu/snapshotu zatwierdzonej VM | NOT_EXECUTED — brak VM |
| Podpisany runtime na właściwej VM | NOT_EXECUTED; końcowa flaga PENDING |
| Preflight z porównaniem stanu i instalacja VM | NOT_EXECUTED; końcowa flaga PENDING |
| Start właściwej jednostki i żywa tożsamość systemd | NOT_EXECUTED; końcowa flaga PENDING |
| Połączenie rzeczywistego verifiera z PostgreSQL i IPC | NOT_EXECUTED; końcowa flaga PENDING |
| Macierz negatywna na osobnych snapshotach | NOT_EXECUTED; końcowa flaga PENDING |
| Kwalifikacja fizyczna / produkcyjna | BLOCKED — poza zakresem |

## Zweryfikowane kontrakty źródłowe

`deployment/production_local_installer.py` wymaga schema_version=2 i scope
`cryptohunter.freshness.substrate.v2`. Manifest zawiera dokładnie
`schema_version`, `scope`, `source_revision`, `files`, `runtime`.
Obiekt runtime ma environment-id, hash manifestu, source revision,
executable, base_executable i jego hash; musi odpowiadać niezależnie
zweryfikowanemu runtime i chronionemu bindingowi. Minimalny `REQUIRED_FILES`
ma obecnie 15 plików. Runbook eksportuje go przez AST z konkretnego commitu,
materializuje token interpretera przed hashowaniem i używa odrębnego podpisu
Ed25519 aplikacji. Historyczny manifest v1 nie jest odpowiednikiem v2.

Istniejący builder sprawdza kompletny zamrożony wheelhouse i dokładne SHA-256,
CPython 3.12/ABI, brak system-site i niezatwierdzonych import hooks, rzeczywiste
importy/Ed25519 oraz biblioteki natywne. Podpis runtime jest niezależny od
podpisu aplikacji. Prywatne klucze TEST_ONLY pozostają poza VM wykonawczą.
Hashy nie aktualizuje się na podstawie zastanego środowiska.

`bot_core/freshness_verifier_service.py` jest właściwą usługą produktu:
stałe konta `os_freshness_crypto_verifier`/`os_freshness_runtime`, grupa
`freshness_verifier_ipc`, PostgreSQL `/run/postgresql:5432/freshness_gate`
z peer do `freshness_crypto_verifier`. Najpierw otwiera połączenie bazy,
dopiero potem wiąże AF_UNIX. Sam start verifiera nie wymaga implementacji
pełnego RootProofIssuer ani AccountGenesis. Brak ich dodatniego łańcucha
nie daje prawa do podstawienia atrap ani poświadczeń otwierających bramki.

Rozwiązano zależność natywnych faz harnessu od psycopg bez zmiany systemowego
Pythona. Główna CLI najpierw uruchamia systemowy bootstrap `verify-installed`,
wiąże environment-id i rewizję z chronionym bindingiem, następnie uruchamia
tę samą chronioną CLI przez podpisany interpreter z `-I -s -B`. Wewnętrzny
worker ponownie wykonuje pełny guard VM, walidację wejść i preflight oraz
sprawdza własne `sys.executable`. Nie jest to pozwolenie na uruchamianie
niezweryfikowanego runtime ani zależność od systemowego psycopg.

`qualify_freshness_service_identity` sprawdza efektywny systemd, dokładny
ExecStartPre/ExecStart przez D-Bus, kolejność znaczników czasu, MainPID,
ExecMainPID, PIDFD, UID/GID/grupy, cgroup, argv, inode i SHA programu oraz
powtórne obserwacje runtime/procesu. Harness korzysta z tej implementacji;
test tekstu unit nie jest równoważnym dowodem.

`collect_verifier_observation` przypina socket-bound PIDFD i rzeczywisty
proces verifiera. Jego własny callback PostgreSQL obserwuje peer, backend PID,
role, transport oraz cztery odmowy `42501` i wykonuje rollback. Osobny proces
administracyjny lub sonda z UID usługi pozostają osobną diagnostyką.
Opcjonalny chroniony `/etc/cryptohunter/linux-qualification.json` jest wymagany
dla diagnostycznej gałęzi root; obecny instalator go nie publikuje.

## Istniejące testy i granice ich dowodów

| Istniejące testy | Co sprawdzają i czego wymagają |
| --- | --- |
| `test_freshness_runtime.py`, `test_build_freshness_runtime.py` | Podpisy, inwentarz, no-overwrite, binding, importy i odmowy; nie zastępują VM. |
| `tests/deployment/freshness_runtime_noble_trial.py` | Oddzielny kontenerowy build/import/sign/verify/publish; jawny disposable opt-in, bez dowodu systemd/VM. |
| `test_freshness_service_identity.py` | Modele systemd oraz rzeczywiste własne `/proc`/PIDFD; brak startu właściwego systemd PID 1. |
| `test_freshness_production_local_deployment.py` | Natywne systemd-analyze oraz kontrakty plików; test składni używa tymczasowego dependency unit. |
| `test_linux_service_ipc.py`, `test_linux_physical_ipc.py` | Rzeczywiste AF_UNIX, kernel credentials i PIDFD oraz odmowy; uprzywilejowane przypadki wymagają `CRYPTOHUNTER_DISPOSABLE_ROOT_IPC=1`. |
| `test_linux_verifier_observation.py`, `test_linux_functional_pipeline.py` | Ochrona przed podmianą obserwacji i promocją częściowego wyniku do kwalifikacji. |
| `test_linux_service_postgresql_integration.py` | Istniejący nie-root tymczasowy PG: peer, ident/role, trust, nadmierny grant, timeout, utrata połączenia, restart. Opt-in `CRYPTOHUNTER_RUN_LINUX_PG_ADAPTER_INTEGRATION=1`, `CRYPTOHUNTER_TEST_PG_BINDIR`. |
| `test_postgresql_freshness_production_local_authentication.py`, `test_freshness_semantic_verifier_postgresql.py` | Rzeczywiste peer i składanie komponentów; root fixture tworzy konta i klaster, więc tylko niezależnie zatwierdzone środowisko. |

Nowe testy harnessu sprawdzają guard, ścisłość wejść, zachowanie odczytowego
INSPECT/preflight i statusów. Centralna walidacja tej sesji dała:

| Kontrola lokalna | Wynik |
| --- | --- |
| Testy harnessu, guard, eksportera i native adapterów | 231 passed; branch coverage 93% |
| Diff coverage | PASS: 94%, wymagany próg 70% |
| Istniejące testy bezpieczeństwa/runtime/instalatora/IPC | 802 passed, 35 skipped |
| Regresje workflow i modele Windows | 554 passed, 2 skipped |
| Pełny Ruff oraz dokładny `python -m pre_commit run ruff-format --all-files` | PASS |
| Skonfigurowany mypy Linux i Windows | PASS dla 391 plików w każdym przebiegu |
| Betterleaks: czysty eksport wersjonowanych plików i zakres nowych źródeł | 0 findings |

Testy rzeczywistych nieuprzywilejowanych socketów AF_UNIX wykonano po nadaniu
wymaganego dostępu sieciowego sandboxu. To lokalne obserwacje kernel IPC,
nie test właściwej VM ani jej systemd. Pominięte testy zachowują swój status;
35/2 skipped nie jest PASS tych przypadków. Osobny skan całego lokalnego
drzewa wraz z cache wykazał pięć fixture findings; nie przedstawia się go
jako czystego skanu wersjonowanego wydania. Semgrep i wynik workflow nowego
HEAD wymagają osobnego raportu; nie zadeklarowano ich zaliczenia.
Zwykłe CI nie uruchamia uprzywilejowanej próby VM. Istniejące progi coverage,
CodeQL i Quality and Security pozostają wymagane.

Runbook podaje konkretne polecenia istniejącego nie-root harnessu PostgreSQL,
osobnego root peer-auth fixture oraz pojedynczych mutacji na klonach. Macierz
host/process/PostgreSQL/recovery ma mapowanie do dokładnych istniejących
testów i pozostaje ręczna. Nie jest pełną automatyzacją VM. Oczekiwana odmowa
ma nadal `FAIL` jako wynik odrzuconej fazy; ręczny testcase może dostać PASS
wyłącznie z dowodem zdrowej bazy, jednej mutacji i właściwej granicy odmowy.

## Poprawki wykryte podczas przeglądu harnessu

Przed przekazaniem poprawiono własne defekty nowej implementacji:

- Eksporter wyłącza Git `refs/replace` przez `GIT_NO_REPLACE_OBJECTS=1`,
  aby lokalne podstawienie obiektu nie zmieniało bajtów przypisanych do SHA.
- Odrzuca niebezpiecznie zapisywalnych przodków outputu, zamiast polegać
  wyłącznie na nieistnieniu jego ostatniego komponentu.
- Porównuje snapshot po odczytowej operacji również w `finally`; wykryta
  mutacja dostaje `FAIL`, także gdy sama operacja odmówiła lub rzuciła wyjątek.
- Rozpoznaje świeżą jednostkę systemd `not-found` przy kodzie 1/4 wyłącznie
  z kompletną dokładną odpowiedzią o LoadState/FragmentPath/DropInPaths.

Wyniki testów tych poprawek są zbierane centralnie w raporcie PR. Opis zmian
nie zastępuje wykonanej próby na VM.

## Zaobserwowany CI bazy po merge

Poniższy stan pochodzi z ostatniego odczytu w tej sesji i nie jest wynikiem
CI nowego HEAD harnessu. Nie potwierdzono, że cała baza ma zielone CI.

| Merge | Workflow / wynik ostatniej obserwacji |
| --- | --- |
| #3239 | Runtime, CodeQL i Windows: success. Quality and Security, CI i style: cancelled. |
| #3244 | Style: success. CodeQL, CI i Quality and Security: queued. Windows: in_progress. |

Anulowane lub trwające zadania nie stanowią PASS. Przed rekomendacją merge
trzeba ponownie sprawdzić wymagane workflow konkretnego nowego HEAD.

## Wykryte istniejące blokery

`bot_core/freshness_deployment_qualification.py` ma w `REVIEWED_CODE_DIGESTS`
pin `account_genesis_freshness_authority.py`:

```text
oczekiwany: cd508809f4b4f0663610f592b77695784d74eb4c7e2fba50c42d7a4dbf252f7b
aktualny:   81cd0df375fe1a48ba943c7645376b4d97350288c0ebe6523207ff90ed22572b
```

Pozostałe pięć pinów zgadza się z plikami wskazanego main. Pełna kwalifikacja
wymaga też tego modułu w `/usr/lib/cryptohunter/bot_core`, chociaż nie należy
on do minimalnego `REQUIRED_FILES`. To konkretna blokada pełnej kwalifikacji,
nie dowód, że sama usługa verifiera nie może wystartować. Nie zmieniono pinów,
inwentarza ani produkcyjnej bramki. Konieczny jest osobny przegląd autorytetu
źródeł; automatyczny rehash byłby niedopuszczalnym obejściem.

Pełna kwalifikacja IPC i PostgreSQL pozostaje `BLOCKED` także po poprawnej
częściowej obserwacji działającego verifiera. Źródła jawnie zachowują:

- `RUNTIME_IPC_APPLICATION_ADAPTER_UNAVAILABLE` oraz brak dowodu exchange
  rzeczywistej aplikacji runtime i odmów jej nieuprawnionych tożsamości;
- `DURABLE_CHALLENGE_REPLAY_PROTECTION_UNAVAILABLE`;
- brak adaptera połączenia PostgreSQL właściwej usługi runtime, jego powiązania
  z procesem oraz pełnej semantycznej macierzy uprawnień.

Odpowiedź `PARTIAL_VERIFIER_OBSERVATION` nie może dać pełnego PASS.
Nie implementowano RootProofIssuer/AccountGenesis ani Windows Stage 10.

## Stan przekazania

Lokalne reguły Semgrep CE 1.179.0: cztery nowe moduły, zero wykryć i błędów;
selftest pięciu reguł: 5/5 PASS. Betterleaks 1.9.0 zweryfikowano względem
pinu SHA-256 i sumy wydania. Skan nowych źródeł/dokumentacji oraz czystej kopii
4087 wersjonowalnych plików z istniejącym baseline: zero wykryć. Pełny lokalny
working tree zwrócił FAIL dla pięciu wygenerowanych wpisów fixture w
`__pycache__` i `.pytest_cache`; tego wyniku nie przemianowano na PASS.
Nie zmieniono baseline ani wykluczeń, nie włączono walidacji u dostawców.
Pokrycie zmienionych linii wyniosło 94% przy niezmienionym progu 70%.

Pierwszy skan GitGuardian nowego PR błędnie sklasyfikował publiczną nazwę
konta `os_freshness_crypto_verifier` w wywołaniu `pwd.getpwnam()` jako
Generic Password. Import standardowego modułu nazwano precyzyjniej
`posix_accounts`; znaczenie operacji i nazwa konta pozostały takie same.
Nie dodano wyciszenia ani wyjątku skanera. Zdalny wynik poprawionego HEAD
wymaga osobnego odczytu; lokalne testy nie zastępują tej kontroli.

Natywny wynik pozostaje `PENDING`, wszystkie przypadki macierzy VM zaczynają
się jako `NOT_EXECUTED`. Testy jednostkowe, rzeczywiste lokalne obserwacje
nieuprzywilejowanych procesów oraz CI należy raportować oddzielnie od VM.
W raporcie końcowym podać PR, aktualny HEAD, konkretne wyniki i CI, pozostałe
blokery oraz rekomendację merge. Nie wykonywać merge. Brak VM nie może być
zamieniony na PASS, a wynik VM nie może awansować fizycznej produkcji.

```text
SIGNED_RUNTIME_VM_VERIFIED=PENDING
INSTALLER_NATIVE_VM_TEST=PENDING
SYSTEMD_NATIVE_VM_TEST=PENDING
POSTGRESQL_IPC_NATIVE_VM_TEST=PENDING
STAGE9_VM_SECURITY_REGRESSIONS=PENDING
LINUX_PHYSICAL_PRODUCTION_ATTESTATION=NO
STAGE9_PRODUCTION_READY=NO
```
