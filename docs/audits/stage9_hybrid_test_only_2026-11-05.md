# Stage 9 — raport weryfikacji hybrydy TEST_ONLY, 5 listopada 2026

**Rekomendacja: REJECT pełnej ochrony FreshnessVerifier przez tę hybrydę.**
Stage 9 pozostaje **IN_PROGRESS / PRODUCTION_NO_GO**,
`STAGE9_PRODUCTION_READY=NO`. Wyłącznie Draft PR, **NO MERGE**.

Baza: `main` po PR #3252,
`5cb848d656c2b7d9d8b06e654be7911f3c9fac0f`.
Początkowy czysty checkout wskazywał starszy `a60be594…`; pobrano main
i utworzono osobną gałąź `test/stage9-hybrid-apparmor-20261105`.
Odczytano AGENTS.md i źródła wymienione w zadaniu, w tym model #3252,
klienta IPC, kwalifikator tożsamości, unit, instalator, bootstrap i entrypoint.
Codebase Memory nie jest dostępne; nie wykonano zapytań ani kontroli pokrycia
indeksu. Wnioski krytyczne potwierdzono bezpośrednio w kodzie.

## Wynik, obejścia i zakres zmiany

[Przegląd platformy](../architecture/stage9_hybrid_platform_review.md)
odtwarza istotne źródła Ubuntu GA `6.8.0-31.31` oraz
`6.8.0-149.149` z archiwum bazowego i diffów porównanych z SHA-256
w .dsc, a także systemd v255 i AppArmor 4.0.1. Zawiera odnośniki,
identyfikatory i granice tego dowodu. Nie odtworzono całego pakietu systemd
Ubuntu, HWE ani konfiguracji rzeczywistej VM.

W obu zbadanych kernelach AppArmor pomija kontrolę profilu ofiary wobec
tracera `unconfined`. Zwykły exec może przywrócić dumpable=1 przed loaderem
i kodem launchera. Późniejsze dumpable=0 i seccomp nie wykluczają wcześniejszej
ingerencji. Samo przejście systemd przez setresuid przed exec nie jest dowodem
osobnej luki: źródła resetu dumpable, bariery i blokady exec również sprawdzono.
**Nie wykonano exploitu**; DAC, ancestry i domyślne Yama ptrace_scope=1 mogą
niezależnie odmówić ataku. Taka odmowa nie dowodzi ochrony AppArmor i nie
zatwierdza nowej podstawy zaufania dla tej propozycji.
Źródła wskazują kandydata do osobnego review samego startu: execute-only ELF
bez odczytu dla UID celu może przez `would_dump` zachować niedumpowalność
przez exec przy `fs.suid_dumpable=0/2`, bez setuid/capabilities. Nie wdrożono
ani nie zakwalifikowano tej konfiguracji; nie rozwiązuje granicy Python.

Pozostają także niezależne drogi lub braki dowodu:

- Kod Python już w pamięci nie wymaga nowego pliku, exec ani PROT_EXEC;
  AppArmor/seccomp nie ustanawiają żądanej granicy interpretowanego kodu.
- Istniejący tracer i jego wcześniejsze modyfikacje nie są usuwane
  retrospektywnie przez zmianę profilu lub późny dumpable.
- Dziecko może dziedziczyć profil i FD. Dozwolone SCM_RIGHTS lub zewnętrzne
  pidfd_getfd mogą zmienić faktycznego handlera, zachowując obserwowane
  poświadczenia socketu. Filtr celu nie filtruje syscalli napastnika.
- Ograniczona mediacja io_uring AppArmor nie jest ogólnym zakazem ringów;
  późny filtr nie cofa istniejących zleceń/mapowań/SQPOLL.
- Wyłączony AppArmor powoduje pominięcie AppArmorProfile w systemd v255;
  complain i ignorowanie błędu nie dają wymaganej odmowy startu.

[Projekt dwóch faz i osiem ocen ACCEPT/REJECT/NOT_PROVEN](../architecture/stage9_hybrid_verifier_test_only.md)
wyjaśnia warunki ochrony oraz granicę klienta.
[Macierz](../runbooks/stage9_hybrid_adversarial_matrix.md) zawiera wszystkie
12 wymaganych scenariuszy, próby uzupełniające i kontrole przypisujące odmowę
do właściwej ochrony.

Nowy prototyp ma wyłącznie wartość diagnostyczną: mały statyczny ELF
instaluje rzeczywisty seccomp w nieuprzywilejowanym procesie potomnym.
Nie tworzy listenera, nie osadza Python, nie przyjmuje danych verifiera.
27 syscalli daje różnicowo EACCES/EUSERS; także ochronę x32 przypisano
różnicowo przez zmianę samej akcji filtra. Błąd instalacji blokuje sondę.
To **NATIVE_SECCOMP_CHILD**, odrębne od pełnej macierzy A+B.

Minimalny profil sondy i projekt unitu pozostają niezaładowane.
Unit ma zawsze fałszywy warunek `ConditionPathIsDirectory=/dev/null`;
nie stanowi gotowego preflight i nie uruchamia usługi.
Nie zmieniono produkcyjnego verifiera, klienta, instalatora, rzeczywistego
unitu, konfiguracji CI ani zamrożonych podpisów, digestów, pinów
i `deployment/stage9_current_status.json`. Brak nowych capabilities,
setuid, brokera root, kluczy i dokumentów. Brak PostgreSQL/AccountGenesis.

## Rzeczywista walidacja

Host: Debian 13.6, Linux 6.18.57 x86-64, UID/GID 1000, PID 1 `tail`.
Nie jest to Ubuntu 24.04/systemd 255 ani zatwierdzona jednorazowa VM.
Python 3.11.17, pytest 9.2.0, GCC 14.2.0, Ruff 0.14.11,
mypy 1.10.0, pre_commit 4.7.0, Semgrep CE 1.179.0, Betterleaks 1.9.0.

| Kontrola | Faktyczny wynik | Granica |
| --- | --- | --- |
| 19 istniejących plików pytest: model seal, cross-UID, VM, IPC, installer/runtime/application/build | **1349 PASS, 4 SKIP**, kod 0 | Modele i nieuprzywilejowane AF_UNIX/PIDFD własnych procesów; bez instalacji lub pełnej hybrydy |
| Nowe testy natywnej sondy | **4 PASS**, kod 0 | 27 odmów wejścia syscalli, różnicowe x32, błąd filtra i błędny tryb; bez rzeczywistego przekazania FD |
| Łącznie unikalne testy | **1353 PASS, 4 SKIP** | Suma dwóch rozłącznych zestawów, bez liczenia powtórzeń |
| Ruff pełnego zakresu quality-security.yml | **PASS** | Bez zmian konfiguracji lub suppressions |
| Ruff nowego testu E,F,I,UP,B | **PASS** | Sonda C dodatkowo budowana `-static -O2 -Wall -Wextra -Werror` |
| Skonfigurowany mypy Linux / win32 | **PASS**, po 392 pliki | Kontrola Windows na Linux; nie natywne wykonanie Windows |
| Jawny mypy nowego testu Linux / win32 | **PASS** | Bez rozszerzania konfiguracji produkcyjnej |
| Dokładny `python -m pre_commit run ruff-format --all-files` | **PASS**, kod 0, bez przeformatowania | Nowy Python dodany do indeksu przed hookiem |
| Semgrep reguł blokujących | **PASS**, 5 reguł, 1123 pliki Python, 0 znalezisk | Reguły repo dotyczą Pythona; nie analizują cBPF/C jako polityki bezpieczeństwa |
| Testy reguł Semgrep | **5/5 PASS** | Doradczy registry baseline NOT_RUN |
| Betterleaks pełnego drzewa | **PASS**, 0 wycieków | Istniejący baseline i piny bez zmian |
| Parser AppArmor Noble 4.0.1 | **PASS**, kod 0 | Statyczna kompilacja z deklarowanym ABI 4.0, bez ładowania lub sprawdzania cech hosta |
| `systemd-analyze verify` hosta 257.13 | **FAIL**, brak ELF pod /opt | Nie instalowano sondy; brak walidacji unitu przez systemd 255, nie jest zaliczona |
| Pełna macierz A+B na Ubuntu, dwa UID, start i rzeczywiste SCM_RIGHTS | **NOT_RUN / BLOCKED** | Brak osobnego zatwierdzenia konkretnej jednorazowej VM |

Cztery istniejące SKIP: brak dodatkowych grup konta, dwa jawne disposable
root harness oraz brak zatwierdzonego wheelhouse Noble. Nie włączano bramek.
Odrębne nowe testy nie wymagają uprawnień administratora ani globalnej polityki.

Pierwszy Betterleaks zgłosił znaną sztuczną fixture klucza w starym
ignorowanym pliku .pyc. Przeniesiono wyłącznie ten cache poza drzewo repo
i powtórzono identyczny skan: zero wycieków. Nie zmieniono reguł lub baseline.
Pierwszy odczyt wersji Semgrep nie mógł zapisać ustawień w home; właściwe
skany użyły katalogu work. Pierwszy systemd-analyze nie mógł użyć SO_PASSCRED
w sandboxie narzędzia; ponowiony z dostępem do lokalnych socketów doszedł
do jawnego ENOENT dla nieinstalowanego ELF. Nie wykonywał startu usługi.
Końcowy powtórny Semgrep w domyślnym sandboxie nie kończył się i został
przerwany (kod 143); powtórzenie seryjne `--jobs 1` z dostępem do lokalnych
socketów przeszło: te same 5 reguł, 1123 pliki, zero znalezisk.

### Statyczny parser bez zmiany polityki jądra

Pobrano z archiwum Ubuntu pakiet
`apparmor_4.0.1really4.0.1-0ubuntu0.24.04.9_amd64.deb`,
SHA-256 `59af23f557d666039708ec176aa056d2fd85125dabd452accf6d86f5c608fb74`,
zgodny z odczytanym Noble-updates Packages.gz. Wyłącznie `dpkg-deb -x`
do katalogu work; nie wykonano maintainer scripts, apt install ani sudo.

Dla `P` wskazującego rozpakowany pakiet i `W` katalog scratch:

```sh
"$P/sbin/apparmor_parser" --version
"$P/sbin/apparmor_parser" \
  --config-file "$P/etc/apparmor/parser.conf" \
  --skip-kernel-load --skip-cache --base "$P/etc/apparmor.d" \
  --policy-features "$P/etc/apparmor.d/abi/4.0" \
  --kernel-features "$P/etc/apparmor.d/abi/4.0" \
  --ofile "$W/stage9-hybrid.bin" \
  "$REPO/deployment/test_only/stage9_hybrid/stage9-hybrid-test-only.apparmor"
```

Parser zgłasza **AppArmor parser version 4.0.1**, kompilacja kod 0.
Plik ABI podany jako kernel-features jest **deklarowanym wejściem**,
nie odczytem aktywnego jądra. Wynik nie dowodzi aktywnego enforce, semantyki
ptrace/unix, poprawnego startu, zgodności całego hosta ani zamknięcia Python.
Nie opiera się na aa-status, SO_PEERSEC lub samej nazwie profilu.

### Odtworzenie testów

Z katalogu repo, w przygotowanym środowisku narzędzi:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
python -m pytest -q -ra \
 tests/scripts/test_stage9_verifier_seal_model.py \
 tests/scripts/test_stage9_cross_uid_directory_contract.py \
 tests/scripts/test_stage9_cross_uid_identity_gates.py \
 tests/scripts/test_stage9_cross_uid_identity_reports.py \
 tests/scripts/test_stage9_cross_uid_permissions.py \
 tests/scripts/test_stage9_cross_uid_trial.py \
 tests/deployment/test_stage9_vm_environment.py \
 tests/deployment/test_stage9_vm_native.py \
 tests/deployment/test_stage9_vm_trial.py \
 tests/deployment/test_prepare_stage9_vm_bundle.py \
 tests/security/test_freshness_runtime_ipc.py \
 tests/security/test_freshness_runtime_ipc_identity.py \
 tests/security/test_linux_service_ipc.py \
 tests/security/test_freshness_service_identity.py \
 tests/deployment/test_linux_physical_ipc.py \
 tests/deployment/test_production_local_installer.py \
 tests/deployment/test_freshness_runtime.py \
 tests/deployment/test_freshness_application.py \
 tests/deployment/test_build_freshness_runtime.py \
 tests/security/test_stage9_hybrid_seccomp_probe.py
python -m mypy
python -m mypy --platform win32
python -m mypy tests/security/test_stage9_hybrid_seccomp_probe.py
python -m mypy --platform win32 tests/security/test_stage9_hybrid_seccomp_probe.py
python -m ruff check --select E,F,I,UP,B tests/security/test_stage9_hybrid_seccomp_probe.py
python -m pre_commit run ruff-format --all-files
semgrep scan --config .semgrep/cryptohunter.yml --metrics=off \
 --disable-version-check --error --jobs 1 bot_core core deployment scripts
semgrep scan --test --config .semgrep/cryptohunter.yml \
 .semgrep/cryptohunter.py --metrics=off --disable-version-check --jobs 1
betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 \
 --no-banner --no-color --report-format json --report-path "$W/secrets.json" .
```

Semgrep wymaga wskazania ustawień/logu do work, jeśli home jest niedostępny.
Ruff pełnego zakresu wykonano dokładnie według jobu
`.github/workflows/quality-security.yml`; nie ograniczano jego listy plików.
Sonda nie może otrzymać pozytywnego wyniku AF_UNIX: nie ma takiego trybu.
Niepowodzenie kompilacji/filtra nie jest automatycznie pomijane.

## CI bazy i zachowane bramki

Pełna paginacja REST, 5 listopada **19:50:57 UTC**,
`commits/SHA/check-runs?filter=latest&per_page=100`:

| Rewizja | PASS | FAIL | SKIP | W toku | Razem |
| --- | --- | --- | --- | --- | --- |
| HEAD PR #3252 `dc88720c…` | 89 | 2 | 13 | 11 | 115 |
| Merge/main `5cb848d…` | 11 | 0 | 1 | 8 | 20 |

To snapshot, nie końcowy zielony wynik. Legacy statuses są puste.
CodeQL i Style preflight miały PASS na obu rewizjach; PR miał także
Lint and Test, mypy, Semgrep/sekrety oraz quality ratchet PASS.

- [Linux deployment](https://github.com/Dudzian/Dudzian/actions/runs/42080555494/job/127452533048):
  FAIL z powodu MISSING_CURRENT_EVIDENCE LIVE_LINUX_INTEGRATION
  dla systemd, clean install i reboot.
- [Windows release gate](https://github.com/Dudzian/Dudzian/actions/runs/42080555494/job/127482311864):
  FAIL z powodu brakujących dowodów reboot/lifecycle/update/failure.
- Klasyczne `branches/main/protection`: **HTTP 403**. Rules/rulesets dały
  puste listy, co nie dowodzi braku klasycznej ochrony. Komplet wymaganych
  checks pozostaje **NOT_PROVEN**.

Odmowy pozostają zachowane. Nie restartowano ani nie dispatchowano workflow.
Fizyczny Linux wymaga osobnego workflow_dispatch i chronionych środowisk;
Stage 10 wymaga jawnego opt-in i właściwych self-hosted runnerów.
Nie wykonano ręcznych testów produkcyjnych. Nowy PR pozostaje Draft,
a bieżące checks trzeba odczytać na jego dokładnym head przed dalszą decyzją.

## Niewykonane i dalszy minimalny krok

Nie wykonano startu pod AppArmor, przełączania enforce/complain/disabled,
uprzywilejowanych prób ptrace, iniekcji, rzeczywistego przekazania listenera
lub połączenia, odziedziczonych ringów/mapowań, pozytywnej komunikacji pełnej
hybrydy, kwalifikacji HWE/systemd255, PostgreSQL ani AccountGenesis.
Nie udzielono zatwierdzenia jednorazowej VM.

Najpierw potrzebna jest decyzja o wymaganiu kodu Python w pamięci i niezależnej
ochronie przed pierwszą instrukcją launchera. Przy zachowaniu wymagań
rekomendowana jest istotna zmiana architektury: mały komponent z domkniętym
kodem i parserem oraz niezależna domena izolacji; jej bezpieczeństwo nadal
wymaga review. Nie należy obchodzić obecnej odmowy executable_sha256.
Dopiero po tych decyzjach minimalną próbą jest osobno zatwierdzona VM
i kontrolowany test wczesnego ptrace z przypisaniem DAC/Yama/AppArmor.
**REJECT pełnej hybrydy, NO MERGE, PRODUCTION_NO_GO.**
