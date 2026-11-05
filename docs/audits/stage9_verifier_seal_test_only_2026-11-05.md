# Stage 9 — raport projektu i modelu verifiera TEST_ONLY

Data: 5 listopada 2026 r. Baza: `main`
`6b0eb5b92d3f083c1ea2a996b7213be6cf348bc2`, merge PR #3251.
**REJECT pełnego wariantu A jako bezpiecznej, zakwalifikowanej tożsamości.**
Stage 9 pozostaje **IN_PROGRESS / PRODUCTION_NO_GO**,
`STAGE9_PRODUCTION_READY=NO`, Draft, `NO MERGE`.

## Wynik i zakres

Powstał niezależny [projekt techniczny z diagramem granicy zaufania](../architecture/stage9_test_only_verifier_seal.md),
[model TEST_ONLY](../../scripts/diagnostics/stage9_verifier_seal_model.py)
oraz [macierz ACCEPT/REJECT](../runbooks/stage9_verifier_seal_adversarial_matrix.md).
Nie powstał wykonywalny bezpieczny launcher ani klient alternatywnie
uwierzytelniający rzeczywistą usługę. Nie można obecnie potwierdzić wymaganej
ochrony startu i zamknięcia aplikacji. Zamiast mechanizmu pozornie bezpiecznego
ograniczono implementację do kompilatora i ewaluatora danych cBPF oraz
syntetycznej sekwencji. Wszystkie jej decyzje mają `MODELED / NOT_RUN`.

Proponowany przyszły launcher natywny osadzałby uprzednio zatwierdzony CPython.
Wymaga wczesnego dumpable=0, oczyszczenia FD także 0–2, zamkniętej inicjalizacji
aplikacji, ponownego zamknięcia pomocniczych FD, NNP oraz nieodwracalnego filtra
o domyślnej odmowie, założonego z TSYNC przed socket/bind/listen. Każdy błąd
kończyłby proces bez fallbacku. Zakaz obejmuje exec, potomków i nowe wątki,
zmianę dumpable/credentials, SCM_RIGHTS, ptrace/pidfd_getfd i io_uring.
Pełna sekwencja i jej warunki są projektem, nie wynikiem wykonania.

Model kompiluje **157 instrukcji cBPF** dla Linux x86_64/LP64, sprawdza arch,
x32 i argumenty wybranych syscalli, koduje `sock_filter` i interpretuje tylko
używany podzbiór instrukcji. Nie zastępuje Linuxowego verifiera BPF.
Model sekwencji obejmuje końcowe zamknięcie FD i zatrzask odmowy; nie modeluje
całego wczesnego startu, rzeczywistego zamykania FD ani mapowań i pracy
io_uring. `succeeded=True` jest wejściem scenariusza, a nie dowodem jądra.
Model klienta porównuje wyłącznie syntetyczne snapshoty; nie wykonuje D-Bus,
procfs, SO_PEERPIDFD ani wymiany protokołu.

Modelowy profil nie jest profilem FreshnessVerifier lub CPython. W szczególności
pełne porównanie `mmap(fd=-1)` może odrzucić legalne wywołanie z inną zawartością
górnego słowa argumentu typu int. Odmowa PROT_EXEC nie dowodzi braku wykonania
Python bytecode ani działania przy READ_IMPLIES_EXEC. Zgodność ABI i aplikacji
nie została zakwalifikowana.

## Przygotowanie i zachowana granica

Zdalny `main` sprawdzono przez API GitHub, następnie pobrano dokładny commit
do osobnej kopii roboczej. Pierwotny checkout wskazywał wcześniejszy merge
`a60be594…`; nie modyfikowano go. Odczytano `AGENTS.md`, dokumentację narzędzi
jakości, propozycję i runbook cross-UID oraz wymagane moduły klienta,
IPC i tożsamości. Sprawdzono też `production_local_installer.py`, bootstrap,
entrypoint, właściwą jednostkę systemd i funkcje uruchomienia verifiera.
Konkretne symbole i konsekwencje wymienia projekt.

Codebase Memory nie jest dostępne w katalogu narzędzi ani lokalnym CLI.
Nie wykonano zapytań grafu, odświeżenia indeksu ani `check_index_coverage`.
To jawna luka przygotowania. Analizę w dostępnych granicach oparto na
bezpośrednim czytaniu źródeł; nie deklaruje się kompletności indeksu.

Dodano wyłącznie pięć nowych plików: projekt, macierz, ten raport, moduł modelu
i jego testy. Poprzednie audyty i blokady pozostają zachowane. Nie zmieniono
produkcyjnego klienta, unitu, instalatora, AccountGenesis, podpisów, digestów,
zamrożonych pinów, konfiguracji CI ani `deployment/stage9_current_status.json`.
Nie dodano capabilities, brokera, kluczy ani zależności produkcyjnych.

## Stan CI po PR #3251

Odczyt wszystkich stron API z 5 listopada, około **17:53 UTC**:

| Rewizja | Checki `filter=latest` | PASS | FAIL | SKIP | W toku | W kolejce |
| --- | --- | --- | --- | --- | --- | --- |
| HEAD PR `bfeec879573a71b9f6ecf2819d1a4762b23056f6` | 116, 2 strony | 96 | 2 | 13 | 5 | 0 |
| Merge/main `6b0eb5b92d3f083c1ea2a996b7213be6cf348bc2` | 48, 1 strona | 18 | 0 | 3 | 15 | 12 |

Legacy statuses: 0 dla obu rewizji. Kontrola historycznych prób
`filter=all` odczytała 218 checków PR na trzech stronach i 48 merge na jednej;
nie liczono ich jako dodatkowych aktywnych bramek. **Końcowy stan CI nie był
dostępny; nie ma podstaw do deklaracji zakończonego zielonego CI.**

- [Linux deployment](https://github.com/Dudzian/Dudzian/actions/runs/42057940652/job/127354417777):
  FAIL, log wskazuje brak LIVE_LINUX_INTEGRATION dla systemd, clean install
  i reboot. Bramka pozostała aktywna.
- [Windows release gate](https://github.com/Dudzian/Dudzian/actions/runs/42057940652/job/127378078458):
  FAIL, brak dowodów reboot recovery, long running lifecycle, update restart
  i failed update safe behavior.
- [CodeQL Python](https://github.com/Dudzian/Dudzian/actions/runs/42057939960/job/127351358072)
  i [JavaScript/TypeScript](https://github.com/Dudzian/Dudzian/actions/runs/42057939960/job/127351357456):
  PASS na HEAD PR.
- Alerty #66 i #67: skonfigurowany odczyt przez API zwrócił **HTTP 403**.
  Stan szczegółowy pozostaje niezweryfikowany; sukces CodeQL nie zastępuje
  odczytu alertu. Nie zmieniano wyjątków skanerów.

Źródła zliczeń to endpoints `commits/SHA/check-runs?filter=latest&per_page=100`,
ich kolejne strony, `filter=all`, `commits/SHA/status` i logi podlinkowanych
jobów. Nie uruchamiano ponownie workflow ani kwalifikacji administracyjnych.

## Faktyczna walidacja lokalna

Środowisko: UID/GID 1000, Debian 13, kernel 6.18.57, Python 3.12.14,
PID 1 inny niż systemd. To środowisko robocze, **nie zatwierdzona VM Noble**.
Narzędzia zainstalowano wyłącznie w lokalnym venv pod `work/`:
Ruff 0.14.11, mypy 1.10.0, Semgrep CE 1.179.0, Betterleaks 1.9.0.

| Kontrola | Faktyczny wynik | Granica dowodu |
| --- | --- | --- |
| Nowy model | **190 PASS**, 0 SKIP; kod 0 | Wyłącznie MODELED; brak wywołań syscalli zabezpieczających |
| Branch coverage nowego modułu | **97,14%**, 116 statements, 24 branches; lokalny próg 70% spełniony | Nie zmieniono żadnego progu repozytorium |
| Pięć istniejących modułów cross-UID i cztery moduły harnessu VM | **466 PASS**, 0 SKIP; kod 0 | Modele, odczyty prywatnych plików i odmowy guardów |
| Pięć istniejących modułów IPC/tożsamości | **383 PASS, 3 SKIP**, kod 0 | Rzeczywiste nieuprzywilejowane AF_UNIX/pidfd/procfs własnych procesów; konta, instalacja, systemd i writer modelowane |
| Łącznie unikalny zestaw pytest | **1039 PASS, 3 SKIP** | Nie dodawano do sumy powtórzenia z coverage |
| Ruff pełnego zakresu Tier A i jawnie `deployment/stage9_vm_environment.py` | **PASS** | Konfiguracja bez zmian |
| Ruff nowego modułu i testów, także `E,F,I,UP,B` | **PASS** | Brak nowych suppressions |
| Skonfigurowany mypy Linux i `--platform win32` | **PASS**, po 392 pliki | Kontrola typów obu platform na Linux, nie natywne wykonanie Windows |
| Jawny mypy nowego modułu Linux/Win32 poza skonfigurowanym zakresem | **PASS**, po 1 pliku | `--follow-imports=skip`, bez zmiany config |
| Dokładny `python -m pre_commit run ruff-format --all-files` | **PASS**, kod 0, bez przeformatowania | Nowe pliki Python dodano do indeksu przed hookiem |
| Semgrep reguły blokujące repozytorium | **PASS**, 5 reguł, 1123 pliki, 0 znalezisk | Istniejące ograniczenia skanowania bez zmian |
| Semgrep końcowego nowego modułu wskazanego jawnie | **PASS**, 1 plik, 0 znalezisk | Skaner dobrał 4 z 5 reguł do tego pliku |
| Testy reguł Semgrep | **5/5 PASS** | Doradczego baseline registry nie pobierano: NOT_RUN |
| Betterleaks całego drzewa oraz zmiany | **PASS**, 0 wycieków | Binarium zgodne z istniejącym SHA-256 CI i checksums wydania |
| Natywna macierz seccomp/start/SCM_RIGHTS/AppArmor/cross-UID | **NOT_RUN** | Brak osobno zatwierdzonej jednorazowej VM; zero przypisanych odmów jądra |
| Kwalifikacja produktu i dowód PostgreSQL | **NOT_RUN / BLOCKED** | Brak połączenia i operacji przygotowania |

Trzy istniejące SKIP dotyczą braku rzeczywistych dodatkowych grup konta
testowego oraz dwóch prób wymagających jawnego jednorazowego harnessu root.
Nie włączano tych bramek. Regresje AF_UNIX uruchomiono z dostępem narzędzia do
lokalnych socketów, nadal jako UID 1000, bez kont/usług i zmian polityki systemu.
Nie są one natywną próbą nowego launchera ani macierzy seccomp.

Pierwsze uruchomienia mypy dały **FAIL narzędziowy**: nowe stubs NumPy 2.5
używały składni nowszej niż skonfigurowany Python 3.11, a 2.2 ujawniło błąd
wewnętrzny mypy 1.10. Lokalnie wybrano zgodne NumPy 1.26.4, następnie oba pełne
zakresy przeszły; nie zmieniono zależności/pinów repozytorium. Semgrep początkowo
nie mógł zapisać ustawień w home; wskazano wyłącznie katalog pod `work/`.
Pierwszy pełny Betterleaks zgłosił fixture w wygenerowanym `.pyc`; usunięto
wyłącznie własne ignorowane cache testów. Ponowny pełny skan dał zero wycieków,
bez zmiany baseline i reguł.

## Odtworzenie

Z root repozytorium, w przygotowanym lokalnym venv:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest -q -ra \
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
  tests/deployment/test_linux_physical_ipc.py
python -m mypy
python -m mypy --platform win32
python -m mypy --follow-imports=skip scripts/diagnostics/stage9_verifier_seal_model.py
python -m mypy --platform win32 --follow-imports=skip scripts/diagnostics/stage9_verifier_seal_model.py
python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui deployment/stage9_vm_environment.py
python -m pre_commit run ruff-format --all-files
semgrep scan --config .semgrep/cryptohunter.yml --metrics=off \
  --disable-version-check --error bot_core core deployment scripts
semgrep scan --test --config .semgrep/cryptohunter.yml .semgrep/cryptohunter.py \
  --metrics=off --disable-version-check --jobs 1
betterleaks dir --baseline-path .betterleaks-baseline.json \
  --redact=100 --no-banner --no-color .
```

Polecenia te odtwarzają kontrole narzędziowe. Nie uruchamiać z nich natywnej
kwalifikacji lub administracyjnych trybów istniejącego harnessu. Katalogi cache
narzędzi należy wskazać w zapisywalnym `work/`, bez zmiany home lub polityki.

## Rekomendacja i nierozwiązane decyzje

**REJECT kontynuowania wariantu A jako implementacji obiecującej wymaganą
ochronę w obecnych założeniach.** Pozostaje wartościowy jako temat dalszych
badań dopiero po osobnym rozstrzygnięciu:

1. Czy dopuszczamy zmianę podstawy tożsamości na zatwierdzony start/ciągłość,
   zamiast niezależnego bieżącego pomiaru kodu; jaka jest granica wobec RCE.
2. Jak chronimy start od chwili exec, także przed aktywnym procesem tego
   samego UID, bez polegania na późnym `TracerPid=0` lub `dumpable=0`.
3. Kto niezależnie zatwierdza pełną closure CPython/ELF/aplikacji i profil
   syscalli oraz dowodzi braku wykonywania niezatwierdzonego kodu Python.
4. Czy AppArmor od startu lub osobny wariant hybrydowy może domknąć brakujące
   warunki. [Porównanie sześciu kryteriów](../architecture/stage9_test_only_verifier_seal.md)
   nie kwalifikuje AppArmor na podstawie samej nazwy profilu lub pakietu.
5. Jakie dokładne bajty, VM i próby operator osobno zatwierdza do niezależnej
   kwalifikacji; jakie przyszłe zmiany zaufanego instalatora będą dopuszczalne.

Wykonanie nie obejmowało administratora, fizycznego laptopa, polityki LSM,
instalacji usługi, publicznego endpointu, AccountGenesis ani PostgreSQL.
Zachowano `RUNTIME_IPC_APPLICATION_ADAPTER_UNAVAILABLE`,
`ACTUAL_RUNTIME_CLIENT_EXCHANGE_UNPROVEN`,
`RUNTIME_UNAUTHORIZED_IDENTITY_DENIAL_UNPROVEN` i
`DURABLE_CHALLENGE_REPLAY_PROTECTION_UNAVAILABLE`.
