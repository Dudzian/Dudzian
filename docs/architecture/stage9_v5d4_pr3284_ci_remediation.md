# PR #3284 — remediacja CI Windows, macOS i Ubuntu

Analizowany HEAD: `5af2363fc60d19a59b3a28a530daaf8ca1c207fe`.
Decyzja wejściowa: **MERGE HOLD**. Ten raport nie udziela niezależnej
aprobaty wyjątku instalacyjnego ani zgody na merge.

## Wszystkie 20 awarii Windows

Przejrzano rzeczywiste tracebacki joba
[Python tests (Windows), 134392982882](https://github.com/Dudzian/Dudzian/actions/runs/44181583687/job/134392982882).
Wynik wejściowy: 20 FAIL, 11405 PASS, 2799 SKIP; pytest zatrzymał się po
20 błędach, więc nie był to kompletny wykaz możliwych regresji.

| Nr | Test bez wspólnego prefiksu `test_` | Przyczyna i poprawka |
| --- | --- | --- |
| 1 | closure: `initializers_transitive_and_deferred_imports` | CP1252 nie koduje `Ć`. Fixture źródłowe i ich odczyty mają jawne UTF-8. |
| 2 | closure: `base_runtime_superset_hashes_cache_and_all_native_bytes` | Profil POSIX używał nieistniejącego `O_NOFOLLOW`. Wczesna kontrolowana odmowa profilu na Windows; pozytywny test wymaga rzeczywistych możliwości POSIX. |
| 3 | closure: `base_runtime_zip_file_root_and_external_symlink_target` | Ten sam brak flagi w odczycie celu symlinku. Zachowano kontrolę całego łańcucha na POSIX. |
| 4 | closure: `base_runtime_bounded_chain_records_every_external_hop` | Brak flagi podczas odczytu zewnętrznego celu. Zachowano limit i wykaz wszystkich ogniw. |
| 5 | closure: `base_runtime_refuses_special_file_without_opening_it` | Brak `mkfifo`; test FIFO ma wąski warunek możliwości systemu. |
| 6 | closure: `base_runtime_unreadable_evidence_refuses[file]` | Brak `O_NOFOLLOW` i nieprzenośne założenie o `chmod`. Test dotyczy POSIX; Windows ma odrębny test odmowy profilu przed I/O. |
| 7 | closure: `base_runtime_unreadable_evidence_refuses[directory]` | Windows `chmod(0)` nie odbiera praw jak POSIX. Nie utożsamiono go z kwalifikacją ACL. |
| 8 | closure: `base_runtime_changed_file_changes_inventory` | Brak `O_NOFOLLOW`; rzeczywisty test zmiany pozostaje na POSIX. |
| 9 | closure: `base_runtime_cli_isolated_without_repository` | Proces potomny kończył się tracebackiem flagi. Niewspierany host kończy się kontrolowanym kodem 2; wspierany profil nadal jest sprawdzany. |
| 10 | dependencies: `archive_executable_bit_does_not_create_unreviewed_native_or_executable_file` | Windows raportuje `0666`, a nie POSIX `0600`. Bajty, hash, manifest, oryginalny ZIP i brak bitów wykonania sprawdzane są wszędzie; dokładny tryb POSIX tylko na POSIX. |
| 11 | environment: `wheel_path_escape[x\\y]` | Fixture była normalizowana przez `ZipInfo` na Windows przed inspekcją. Nowa fixture zachowuje rzeczywiste bajty obu nagłówków ZIP. Walidator nadal odrzuca backslash. |
| 12 | environment: `runtime_unexpected_fifo_refuses_before_open` | Brak `mkfifo`; wąski warunek API dla rzeczywistego FIFO. Przenośne kontrole manifestu, symlinków i odmów pozostają aktywne. |
| 13 | environment: `native_gate_checks_receipt_and_never_invokes_preflight` | Fixture próbowała mierzyć Ubuntu na innym systemie. Pozytywna bramka wymaga Linux; osobne testy rzeczywistego Windows/macOS i modeli odmowy nie czytają `/etc/os-release`. |
| 14 | environment: `staging_copies_without_loading_code` | Fixture modelowała Linux, ale rzeczywisty Windows nie ma `geteuid`. Staging wymaga Linux; builder sprawdza system i dostępność API przed jego użyciem. |
| 15 | environment: `build_parses_exact_pinned_abi_bytes_once[abi_request]` | Ten sam Linux-only staging. Ochrona przypiętych bajtów pozostaje bez zmian. |
| 16 | environment: `build_parses_exact_pinned_abi_bytes_once[abi_expectations]` | Ten sam Linux-only staging i kontrolowana odmowa poza profilem. |
| 17 | environment: `real_verification_to_target_omits_only_reviewed_cache` | Ten sam brak granicy systemu w fixture; pełna ścieżka materializacji nadal testowana na Linux. |
| 18 | environment: `actual_build_requires_complete_explicit_omissions[missing]` | Ten sam staging. Zachowano odmowę niekompletnej aprobaty w builderze oraz przenośne testy verifiera. |
| 19 | environment: `actual_build_requires_complete_explicit_omissions[incomplete]` | Jak wyżej; nie wyłączono wymagania pełnego wykazu. |
| 20 | environment: `actual_build_requires_complete_explicit_omissions[extra]` | Jak wyżej; dodatkowe pominięcie nadal jest odmową. |

Nie wyłączono żadnego całego modułu testowego. Test mutacji deskryptora
`os.fdopen` dotyczy teraz jawnie POSIX. Windows ma rzeczywisty test odmowy
odczytu przy istniejącym uchwycie zapisującym, zamiast pozorowania POSIX API.

## macOS i Ubuntu

Jedyny FAIL macOS w
[jobie 134392983178](https://github.com/Dudzian/Dudzian/actions/runs/44181583687/job/134392983178)
to pozycja 13. Builder, pomiar hosta i launcher odmawiają poza Linux
przed użyciem API Ubuntu; brak wymaganych API również daje kontrolowaną
odmowę. Nie udaje się Ubuntu na rzeczywistym macOS.

Jedyny FAIL Ubuntu w
[jobie 134392983244](https://github.com/Dudzian/Dudzian/actions/runs/44181583687/job/134392983244)
wynikał z nieaktualnego kontraktu jobów. Dokładna równość zbiorów pozostaje;
dodano wyłącznie `stage9-v5d4-static-environment`. Odrębny ścisły test
sprawdza jego warunek dispatch, read-only permissions, timeout, brak
produkcyjnego environment, checkout dokładnej rewizji z historią,
syntetyczny zakres testów i wymagane artefakty. Workflow nie zmieniono.
Regresja lokalna: przed poprawką 1 FAIL / 37 PASS; po poprawce 39 PASS.

## Odczyt przenośny a dowody POSIX

Pełny wykaz bazowego runtime zachowuje semantykę POSIX i wymaga rzeczywistych
flag `O_NOFOLLOW` i `O_NONBLOCK`. Brak możliwości daje
`BASE_RUNTIME_POSIX_CAPABILITY_REQUIRED` przed dostępem do korzeni.
Przenośna inspekcja AST, wheel i statycznych danych ABI pozostaje dostępna.

Trzy czytniki zwykłych plików nie stosują już zerowego fallbacku dla
`O_NOFOLLOW`. POSIX wymaga obu flag. Windows używa wspólnego
`windows_read_regular()` z zaufanego sąsiedniego modułu dependencies:

- `CreateFileW` z `OPEN_REPARSE_POINT`, wyłącznie istniejący lokalny plik;
- przypięcie każdego rodzica bez współdzielenia usuwania;
- kontrola natywnych atrybutów, typu dyskowego i finalnej ścieżki wszystkich uchwytów;
- odrzucenie reparse points, katalogu zamiast pliku i wielu hardlinków;
- brak współdzielenia zapisu/usuwania pliku podczas odczytu;
- limit rozmiaru oraz natywna tożsamość i metadane przed/po odczycie;
- `ReadFile` zamiast niepełnej tożsamości CRT `fstat` starszego Pythona.

Wzorzec pochodzi z repozytoryjnych czytników Windows custody, ale helper
korzysta wyłącznie ze standardowej biblioteki i nie importuje pywin32,
cryptography ani instalatora. Nie kwalifikuje ACL, custody lub hosta.
Odczyt danych nie wykonuje żadnego kodu dystrybucji i nie upoważnia do budowy
środowiska Ubuntu na Windows. Natywne testy Windows sprawdzają trzy czytniki,
Unicode/bajty binarne, realne hardlinki/symlinki, brak API, blokadę writerów
i limity. Nie symulują `os.name` jako dowodu działania Windows.

## Niezmienione blokady i walidacja

Porównanie AST potwierdza niezmienione: `TARGET`, `BYTECODE_POLICY_ID`,
`BYTECODE_POLICY`, inspektor wheel/RECORD, generator i verifier propozycji,
oczekiwania pominięć oraz kontrolę kolizji. Nie zmieniono niezależnej aprobaty,
62 kolizji PySide6, właściwych kontroli ABI, `pyproject.toml`, workflow,
pre-commit, reguł Semgrep ani bazy skanera sekretów.

[linux-deployment-integration, 134400965589](https://github.com/Dudzian/Dudzian/actions/runs/44181583524/job/134400965589)
pozostaje prawidłową odmową bez aktualnych dowodów fizycznych:

```text
LINUX_SYSTEMD_INTEGRATION=MISSING_CURRENT_EVIDENCE
LINUX_CLEAN_INSTALL=MISSING_CURRENT_EVIDENCE
LINUX_REBOOT_LIFECYCLE_UPDATE=MISSING_CURRENT_EVIDENCE
LINUX_PRODUCTION_READY=NOT_READY
```

Nie zmieniono exit code, warunków joba ani wymagań dowodowych. Nie dodano
`continue-on-error`, nie wytworzono dowodów i nie uruchomiono fizycznego
kontrolera ani VM. Końcowy nowy SHA i rzeczywiste wyniki PASS/FAIL/SKIP
Windows, macOS i Ubuntu zostaną zapisane w opisie istniejącego PR #3284;
wyniki lokalne nie zastępują natywnych Actions.

Lokalnie po poprawkach: **V5d4 377 PASS / 22 SKIP** oraz **Stage 9 safe CI
39 PASS**. Precyzyjne SKIP na Linux obejmują 19 natywnych przypadków Windows,
jeden natywny test odmowy profilu Windows i dwa testy CLI wymagające hosta
innego niż Linux. Pełny dotychczasowy zakres Ruff, dokładny hook
`python -m pre_commit run ruff-format --all-files`, configured mypy Linux
i Win32, wszystkie hooki pre-commit, blokujący Semgrep i jego pięć testów,
Betterleaks oraz pip-audit kanonicznego locka zakończyły się PASS.
Nie obniżono progów, zakresów ani limitów.

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```
