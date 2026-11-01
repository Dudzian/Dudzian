# Stage 9 Linux — odtwarzalna integracja adapterów w izolowanej VM

Ta procedura sprawdza rzeczywiste mechanizmy Linux i PostgreSQL oraz kod
adapterów. Nie jest fizyczną kwalifikacją Della ani instrukcją wdrożenia.
Nie wykonujemy jej automatycznie na Dellu. Nie uruchamia instalatora produktu,
nie restartuje hosta i nie przyjmuje kluczy ani danych produkcyjnych.
Architekturę i ograniczenia opisuje
[mapa komponentów](../architecture/stage9_linux_functional_adapters.md).

## Przygotowanie odtwarzalnego środowiska

Utwórz nową, jednorazową VM Ubuntu 24.04 z systemd, cgroup v2, jądrem
obsługującym `SO_PEERPIDFD` i PostgreSQL 16 lub nowszym. Wykonaj snapshot
przed testami. VM nie może mieć zamontowanych katalogów Della, produkcyjnych
socketów, baz, urządzeń custody, kluczy ani danych użytkowników. Zapisz obraz
systemu, wersję jądra i dokładny SHA testowanego PR. Brak wymaganej funkcji
jądra jest rzeczywistą luką środowiska, a nie powodem do wyłączenia kontroli.

Poniższe polecenia przygotowawcze wykonuje operator wyłącznie w tej VM:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv build-essential libpq-dev \
  postgresql-16 postgresql-client-16 util-linux systemd
sudo install -d -m 0755 -o "$(id -un)" -g "$(id -gn)" \
  /opt/cryptohunter-stage9-test
git clone https://github.com/Dudzian/Dudzian.git /opt/cryptohunter-stage9-test
cd /opt/cryptohunter-stage9-test
git fetch origin codex/stage9-linux-functional-adapters
git checkout --detach FETCH_HEAD
git rev-parse HEAD
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]' \
  'ruff==0.14.11' 'mypy==1.10.0' pre-commit
mkdir -p work/stage9-vm-results
```

Przed wykonaniem porównaj wydrukowany SHA z zatwierdzonym SHA PR. Samo
pobranie nazwy gałęzi nie przypina jej przyszłych zmian. Zapisz SHA w wynikach.
Nie uruchamiaj `deployment/install-production-local.sh`; testowe klastry
opisane poniżej są odrębne od domyślnego klastra pakietu PostgreSQL.

## Regresja bez uprawnień administratora

Jako zwykły użytkownik VM wykonaj:

```bash
cd /opt/cryptohunter-stage9-test
.venv/bin/python -m pytest -q \
  tests/security/test_linux_service_ipc.py \
  tests/security/test_linux_service_postgresql.py \
  tests/deployment/test_linux_physical_ipc.py \
  tests/deployment/test_linux_physical_postgresql.py \
  tests/deployment/test_linux_physical_evidence.py \
  tests/deployment/test_linux_physical_remediation.py \
  tests/deployment/test_linux_physical_native.py \
  tests/deployment/test_linux_physical_collector.py \
  tests/deployment/test_linux_release_identity.py \
  tests/security/test_freshness_semantic_verifier.py \
  tests/security/test_freshness_production_local_deployment.py \
  tests/architecture/test_m05_account_genesis_high_level_freshness_authority_runtime_wiring.py \
  tests/architecture/test_m05_account_genesis_freshness_crypto_to_postgres_cas_authorization_contract.py \
  tests/security/test_postgresql_freshness_authority_acl_contract.py \
  --junitxml=work/stage9-vm-results/bez-uprawnien.xml
```

Testy wymagające root mogą tu zostać pominięte. Zapisz każde pominięcie;
nie oznaczaj całej macierzy integracyjnej jako zaliczonej. Sprawdzenie
jednostki przez `systemd-analyze verify` kontroluje plik, a nie dowodzi
uruchomienia tej jednostki ani jej skutecznego sandboxu.

Nowy test PostgreSQL uruchamia **świeży, tymczasowy klaster** i nie tworzy
kont systemowych. Działa jako nieuprzywilejowany użytkownik. Włącz go jawnie:

```bash
cd /opt/cryptohunter-stage9-test
CRYPTOHUNTER_RUN_LINUX_PG_ADAPTER_INTEGRATION=1 \
CRYPTOHUNTER_TEST_PG_BINDIR=/usr/lib/postgresql/16/bin \
  .venv/bin/python -m pytest -q \
  tests/security/test_linux_service_postgresql_integration.py \
  --junitxml=work/stage9-vm-results/postgresql-adapter.xml
```

Ten test mapuje bieżącą tożsamość systemową na rolę verifiera w prywatnym
klastrze. Dowodzi działania granic peer/SQL/odmów adaptera; nie dowodzi
połączenia zainstalowanej usługi produktu. Nie zastępuj ścieżki binariów
parametrem DSN prowadzącym do cudzej bazy. Nie ustawiaj `PGPASSWORD`,
`PGSERVICE` ani innych `PG*`; adapter odrzuca niejawne ustawienia libpq.

## Uprzywilejowane testy rzeczywistych procesów i peer

Istniejące `isolated_peer_cluster` i `linux_processes` mają własne kontrole
Linux/root, lecz nie mają pełnej blokady uniemożliwiającej wykonanie na
przypadkowym fizycznym hoście. Wymagają jawnego wyboru przez operatora.
Poniższy strażnik jest częścią procedury, nie nowym uprawnieniem produktu.

W tej **jednorazowej VM** operator najpierw tworzy znacznik:

```bash
sudo install -o root -g root -m 0600 /dev/null \
  /run/cryptohunter-stage9-disposable-vm
```

Następnie wykonuje cały blok z kontrolą VM i znacznika:

```bash
sudo bash --noprofile --norc <<'SH'
set -euo pipefail
systemd-detect-virt --vm
test -f /run/cryptohunter-stage9-disposable-vm
test ! -L /run/cryptohunter-stage9-disposable-vm
test "$(stat -c '%u:%g:%a' /run/cryptohunter-stage9-disposable-vm)" = '0:0:600'
for principal in os_freshness_crypto_verifier os_freshness_runtime os_freshness_outsider; do
  if getent passwd "$principal" >/dev/null; then
    echo 'VM nie jest świeża: konto fixture już istnieje' >&2
    exit 1
  fi
done
cd /opt/cryptohunter-stage9-test
CRYPTOHUNTER_DISPOSABLE_ROOT_IPC=1 .venv/bin/python -m pytest -q \
  tests/security/test_linux_service_ipc.py \
  tests/security/test_production_signer_ipc.py \
  tests/security/test_production_signer_linux_processes.py \
  tests/security/test_postgresql_freshness_production_local_authentication.py \
  tests/security/test_freshness_semantic_verifier_postgresql.py \
  --junitxml=work/stage9-vm-results/uprzywilejowane.xml
SH
```

Fixture PostgreSQL tworzy konta `os_freshness_crypto_verifier`,
`os_freshness_runtime`, `os_freshness_outsider`, własny katalog
`freshness-peer-gate-*`, HBA/ident i klaster z wyłączonym TCP oraz włączonymi
fsync/synchronous_commit. Zakończenie sprząta własny klaster i utworzone konta.
Testy restartują **ten odizolowany PostgreSQL**, nie host ani klaster Della.
Integracja freshness sprawdza rzeczywisty resolver/verifier/preparer,
proces runtime, exact replay, utratę odpowiedzi COMMIT i odzyskanie po
restartach. Część fixture ma testowe obiekty schematu i klucze; nie są
wydaniem produkcyjnym ani dowodem zatwierdzonej instalacji.

Fixture signerów używa liczbowych tożsamości testowych, procesów i własnych
socketów pod `/run/cryptohunter-rpi02-test-*`. Sprawdza realne DAC, pidfd,
odmowy i restart procesu. Nie dowodzi produkcyjnego custody, systemd ani
gotowości Source/CCA. Nie podłączaj do niego prawdziwych credentiali.

Nowy pozytywny test wymiany aplikacyjnej root ma dodatkowy warunek
`CRYPTOHUNTER_DISPOSABLE_ROOT_IPC=1`; ustawienie go bez uprawnień root nie
usuwa pozostałych kontroli. Test odmowy rzeczywistej tożsamości nie-root
wykonuje się w poprzednim, nieuprzywilejowanym etapie. Oba etapy są potrzebne.

Po awarii teardown zachowaj log i snapshot VM. Nie usuwaj zasobów po samej
podobnej nazwie i nie uruchamiaj poleceń sprzątania na innym hoście. Całą VM
można odtworzyć ze snapshotu po zachowaniu wyników.

## Regresja Source/CCA i bramki jakości

Zwykłym użytkownikiem uruchom istniejące testy z zastąpionymi granicami systemu:

```bash
.venv/bin/python -m pytest -q \
  tests/deployment/test_stage9_trusted_installation_contract.py \
  tests/deployment/test_stage9_trusted_installation_files.py \
  tests/deployment/test_stage9_trusted_installation_runtime.py \
  tests/deployment/test_stage9_trusted_installation_identity_gate.py \
  tests/deployment/test_stage9_native_linux_source_cca_preflight.py \
  tests/security/test_stage9_source_cca_test_lifecycle.py \
  tests/security/test_stage9_source_cca_test_probes.py \
  tests/security/test_stage9_source_cca_test_rootless.py \
  tests/security/test_source_cca_cryptographic_chain_linux.py \
  --junitxml=work/stage9-vm-results/source-cca.xml
.venv/bin/python -m pre_commit run ruff-format --all-files
.venv/bin/python -m mypy
.venv/bin/python -m mypy --platform win32
```

Wykonaj również dokładny zakres Ruff i testów obowiązujący w aktualnym
`Quality and Security`. Nie zmieniaj progów pokrycia, wyłączeń ani zakresów.
Wynik mypy z `--platform win32` sprawdza typy; nie zastępuje uruchomienia
testów na Windows. macOS i Windows wymagają osobnych wyników natywnych CI.
CodeQL i pełne `Quality and Security` muszą dotyczyć dokładnego SHA PR.
Stan oczekujący, pominięty i zakończony błędem pozostają oddzielnymi statusami.

## Odczytowy adapter we właściwej usłudze

Opcjonalny plik `/etc/cryptohunter/linux-qualification.json` ma dokładnie:

```json
{"schema_version":1,"installation_id":"<64 małe znaki szesnastkowe>"}
```

Właściwy identyfikator wyprowadza się z zatwierdzonej instalacji; placeholder
nie jest poprawną konfiguracją. Wymagane są root:root, tryb `0644`, chronione
katalogi nadrzędne oraz brak symlinków i rozszerzonych ACL. Ten plik włącza
odczytową gałąź istniejącego verifiera; nie tworzy runtime ani autorytetu
wydania. Plik jest czytany przy konstrukcji usługi. Ręczny restart verifiera
do takiego testu jest dopuszczalny wyłącznie w wydzielonej VM i wymaga
uprzednio przygotowanej testowej instalacji; ta procedura nie uruchamia go
automatycznie. Nie kopiuj pliku na Della w ramach tego PR.

## Wymagany zapis wyniku

Zachowaj SHA źródeł, wersje Pythona/jądra/PostgreSQL, JUnit, pełne logi,
liczbę wykonanych i pominiętych przypadków, nazwę obrazu VM oraz obserwacje
przed i po teście. Surowe dane muszą pozwalać odróżnić `42501` od innego
błędu, rzeczywistą odmowę aplikacji od timeoutu i nowy proces od ponownie
użytego PID. Nie zachowuj wartości credentiali ani danych prywatnych tabel.

Wyniki tej procedury oznaczają integrację adapterów w VM. Pełna fizyczna
kwalifikacja nadal wymaga zatwierdzonego wydania, obu rzeczywistych usług,
całego łańcucha challenge i dowodów siedmiu faz. Zachowaj
`LINUX_PRODUCTION_READY=NO`; nie zmieniaj Stage 9 na COMPLETE.

### Zebranie częściowej diagnostyki bez instalacji i restartu

Profil instalacji może zawierać opcjonalne pole:

```json
{"functional_qualification":{"installation_id":"<niezależny identyfikator 64hex>"}}
```

W `configuration` musi jednocześnie przypinać dokładne bajty
`/etc/cryptohunter/linux-qualification.json` i jego metadane root:root/0644,
a `ipc` zawierać właściwy `/run/cryptohunter/freshness/verifier.sock`.
Identyfikator jest niezależny od hasha dokumentu, aby uniknąć cyklicznego
haszowania konfiguracji. Zatwierdzony challenge wiąże cały kontrakt, źródła,
artefakty i konfigurację.

Na już przygotowanym, zatwierdzonym hoście kwalifikacyjnym operator może
wybrać odczytową diagnostykę zamiast faz mutujących:

```bash
python -m deployment.linux_physical_collector --observe-verifier \
  --challenge /chroniony/challenge.json --manifest /chroniony/installation.json \
  --private-key /chroniony/producer.key --bundle /chroniony/nowa-diagnostyka.json
python -m deployment.linux_physical_evidence verify-diagnostic \
  --policy /chroniony/policy.json --expected-policy-sha256 '<zatwierdzony digest>' \
  --source-revision '<zatwierdzony SHA>' --ci-run-id '<run>' --ci-run-attempt '<attempt>' \
  --challenge /chroniony/challenge.json --bundle /chroniony/nowa-diagnostyka.json \
  --output /chroniony/wynik-diagnostyki.json
```

Należy użyć dedykowanego klucza producenta, nigdy kluczy produkcyjnych.
Pierwsze polecenie wymaga root oraz rzeczywistego hosta spełniającego istniejącą
politykę fizyczną. Nie jest poleceniem do automatycznego wykonania na Dellu.
Weryfikator zwraca `PARTIAL_BLOCKED`; osobna domena podpisanego dokumentu nie
może zostać użyta jako siedem faz bundle ani wynik platform readiness.
Sam odczyt diagnostyki nie zużywa produkcyjnego challenge. Trwały rejestr
zużycia pełnej kwalifikacji i blokady brakującego runtime pozostają wymagane.
