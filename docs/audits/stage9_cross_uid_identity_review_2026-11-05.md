# Stage 9 — przegląd tożsamości verifiera między UID, 5 listopada 2026 r.

Wynik: **BLOCKED / decyzja architektoniczna do zatwierdzenia**.
Podstawa: `main` `a60be5946a02cfad317ec895b2f04ef7f63407e2`, po scaleniu
[PR #3250](https://github.com/Dudzian/Dudzian/pull/3250).
Nie zmieniono produkcyjnego uwierzytelniania.

## Rozstrzygnięcie

Przy odrębnych UID zwykły runtime standardowo nie może odczytać
`/proc/PID/exe` verifiera: obowiązuje `PTRACE_MODE_READ_FSCREDS`, a grupa IPC
nie usuwa kontroli UID. Dostępne stat22, cgroup i cmdline nie zastępują tego
dowodu. SO_PEERCRED i SO_PEERPIDFD dotyczą zapisanego procesu listenera,
niekoniecznie odbiorcy odziedziczonego lub przekazanego deskryptora.
Exec zachowuje PID, start-time i pidfd. MainPID, ExecStart oraz podpisany
inwentarz nie dowodzą bieżącego kodu aplikacji.

[Propozycja architektury](../architecture/stage9_cross_uid_verifier_identity_proposal.md)
zawiera źródła Linux 6.8/systemd 255, analizę execve/execveat/fork/SCM_RIGHTS,
tabelę trzech wariantów i wymagania ich kwalifikacji. Preferowany kierunek
to zatwierdzony start i nieodwracalne zamknięcie procesu przed listen,
bez nowych capabilities ani uprzywilejowanego brokera. Akceptacja musi jawnie
objąć zmianę podstawy zaufania i granicę aplikacji Pythona. Jeżeli wymagany
pozostaje niezależny pomiar bieżącego kodu, bez zmiany tej granicy wynik
pozostaje BLOCKED.

## HEAD, instrukcje i CI

Odczytano `AGENTS.md`. Przed analizą zależności sprawdzono dostępność Codebase
Memory w narzędziach, zasobach/szablonach MCP, lokalnym CLI i katalogu
integracji. Serwer nie był udostępniony. Nie wykonano zapytań grafu,
odświeżenia ani check_index_coverage; to niespełniony element przygotowania.
Zależności i wnioski bezpieczeństwa sprawdzono bezpośrednio w źródłach.

Stan CI podstawy odczytany 5 listopada o 09:24 UTC:

| Przebieg / kontrola | Rzeczywisty stan |
| --- | --- |
| [Style preflight](https://github.com/Dudzian/Dudzian/actions/runs/41999625336) | PASS |
| [CI po push main](https://github.com/Dudzian/Dudzian/actions/runs/41999625783) | Niezakończony; Python tests (Ubuntu) czeka. Lint and Test oraz Mypy (core packages) PASS. |
| [Quality and Security](https://github.com/Dudzian/Dudzian/actions/runs/41999625670) | W toku; Semgrep and working-tree secrets PASS. |
| [Windows Build](https://github.com/Dudzian/Dudzian/actions/runs/41999625398) | W toku |
| CodeQL, Marketplace Catalog, Stress Lab Report | PASS |

Nie deklaruje się zielonego kompletu CI po #3250.

## Wykonana walidacja lokalna

Środowisko: nieuprzywilejowany UID 1000, Debian 13, Python 3.12, kernel
6.18.57, PID 1 inny niż systemd. Narzędzia zainstalowano w lokalnym venv
pod `work/`, bez instalacji systemowej i zmian zależności repozytorium.
Ruff 0.14.11, mypy 1.10.0, Semgrep CE 1.179.0.

| Kontrola | Wynik i granica dowodu |
| --- | --- |
| Dziewięć istniejących plików pytest wskazanych poniżej | **654 PASS, 3 SKIP**, kod 0. Rzeczywiste lokalne sockety/pidfd; konta, instalacja, systemd i writer pozostają jawnie modelowane. |
| Dwa testy związane z #3250, pięć powtórzeń | **10/10 PASS**, każdy przebieg kod 0: poprawna wymiana semantic oraz zakończenie potomka ignorującego SIGTERM. |
| `python -m mypy` i `python -m mypy --platform win32` | **PASS**, po 392 pliki, siedem uwag informacyjnych o ciałach funkcji bez typów. |
| Nowe testy integralności raportów | **11 PASS**, kod 0; wyłącznie format i dziesięć odmów uszkodzonych raportów. Nie stanowią próby UID ani jądra. |
| Nowy harness: `mypy --follow-imports=skip` | **PASS**, jeden plik sprawdzony jawnie poza skonfigurowanym zakresem. |
| Ruff pełnego zakresu Tier A i nowego harnessu/testów | **PASS**, bez zmian konfiguracji. |
| Dokładny `python -m pre_commit run ruff-format --all-files` | **PASS**, kod 0, bez zmian drzewa roboczego; sprawdzono po dodaniu nowych plików do indeksu. |
| Semgrep: reguły blokujące repozytorium | **PASS**, pięć reguł, zero znalezisk. |
| Semgrep nowego harnessu wskazanego osobno | **PASS**, zero znalezisk; skrypt został rzeczywiście przeskanowany. |
| Testy reguł Semgrep | **5/5 PASS**. Nie wykonywano pobrania doradczych reguł społecznościowych. |
| Betterleaks 1.9.0: pięć plików zmiany | **PASS**, zero wycieków; binarium sprawdzono względem istniejącego SHA-256 CI, bez zmiany pinu lub baseline. |

Lokalne CLI harnessu: `--help` zwraca kod 0; `probe` zwraca kod 2/BLOCKED
z powodu braku systemd jako PID 1; `preflight` zwraca kod 2/BLOCKED
`EXECUTION_REQUIRES_VM_ADMINISTRATOR`. Nie wykonano trybu listener/snapshot.

Pierwszy pytest w domyślnym sandboxie dał 562 PASS / 50 SKIP / 45 FAIL
z powodu EPERM lokalnych AF_UNIX/SO_PEERPIDFD. Ten sam zestaw przeszedł po
zezwoleniu narzędziu na lokalne sockety, bez zmian kodu, guardów lub oczekiwań.
Pozostałe trzy pominięcia to brak dodatkowych grup konta testowego oraz dwa
testy wymagające jawnego jednorazowego harnessu root. Nie włączano tych bramek.

Odtworzenie zestawu testów z root repo, w przygotowanym środowisku narzędzi:

```sh
python -m pytest -q -ra \
  tests/security/test_freshness_runtime_ipc.py \
  tests/security/test_freshness_runtime_ipc_identity.py \
  tests/security/test_linux_service_ipc.py \
  tests/security/test_freshness_service_identity.py \
  tests/deployment/test_linux_physical_ipc.py \
  tests/deployment/test_stage9_vm_trial.py \
  tests/deployment/test_stage9_vm_environment.py \
  tests/deployment/test_stage9_vm_native.py \
  tests/deployment/test_prepare_stage9_vm_bundle.py
python -m mypy
python -m mypy --platform win32
python -m pytest -q tests/scripts/test_stage9_cross_uid_identity_reports.py
python -m mypy --follow-imports=skip scripts/diagnostics/stage9_cross_uid_identity.py
python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
python -m pre_commit run ruff-format --all-files
semgrep scan --config .semgrep/cryptohunter.yml --metrics=off \
  --disable-version-check --error bot_core core deployment scripts
semgrep scan --test --config .semgrep/cryptohunter.yml .semgrep/cryptohunter.py \
  --metrics=off --disable-version-check --jobs 1
```

## Próba wymagająca VM i pozostawione blokady

Nowy [harness odmowy](../runbooks/stage9_cross_uid_native_identity.md)
nie tworzy kont/usług/kluczy ani bazy. Operator po osobnym zatwierdzeniu
przygotowuje jednorazową Ubuntu Server 24.04 z rzeczywiście odrębnymi UID/GID.
Istniejąca bramka approval wiąże konkretną VM, boot, commit, pakiety i izolację.
Niezmieniony klient musi odmówić przy rzeczywistym braku exe, a listener
wykazać EOF i zero bajtów. Obserwacyjny ślad Pythona dowodzi wejścia i wyjątku
rzeczywistej kontroli serwera; nie zastępuje obserwacji jądrowych.

**VM: NOT_RUN** — w tej sesji nie ma zatwierdzonej VM. Listener nie ma operacji
przygotowania ani połączenia SQL, więc ta próba nie może wykonać przygotowania.
Dowód niezmienności rzeczywistego PostgreSQL pozostaje **NOT_RUN/BLOCKED**,
zgodnie z pełnym kontraktem ACCEPT/REJECT. Dodatnia wymiana właściwej usługi
i testy egzekwowania exec/fork/delegacji wymagają osobnego prototypu,
zatwierdzenia architektury i wykonania VM.

Pozostają cztery blokady IPC, integracja AccountGenesis, trwały replay i pełna
kwalifikacja produktu. Odczytowe porównanie pinów potwierdziło pięć zgodnych
`REVIEWED_SUBSTRATE_DIGESTS` oraz pięć zgodnych `REVIEWED_CODE_DIGESTS`;
AccountGenesis nadal ma rozbieżność z szóstym pinem. Niczego nie aktualizowano.
Nie zmieniono status JSON, CI ani progów. Nie wykonano operacji administracyjnych
na komputerze użytkownika, usług produkcyjnych lub fizycznym Dellu.
**STAGE9_PRODUCTION_READY=NO; Draft, NO MERGE.**
