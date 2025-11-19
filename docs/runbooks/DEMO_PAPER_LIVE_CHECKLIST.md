# Runbook: Checklista przejścia Demo → Paper → Live

## Cel
Zapewnienie, że każda promocja środowiska tradingowego spełnia wymagania techniczne, ryzyka i compliance zanim przejdziemy z trybu demo do paper, a następnie do live. Dokument jest przeznaczony dla operatorów runtime, zespołu ryzyka oraz compliance i stanowi obowiązkową listę kontrolną przed zmianą środowiska.

## Etap 1 – Gotowość demo (pre-check)
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Zweryfikuj konfigurację `config/core.yaml` poleceniem `python scripts/validate_config.py --profile demo` (flaga `--profile`/`-p` ogranicza kontrolę do wskazanego typu środowiska) | Inżynier Runtime | Raport walidacji, `config/core.yaml` (hash SHA-256) | Raport bez błędów, hash wpisany do decision logu |
| 2. Uruchom `python scripts/run_metrics_service.py --shutdown-after 0 --jsonl logs/metrics_demo.jsonl` i zweryfikuj TLS/mTLS | Observability | `logs/metrics_demo.jsonl`, snapshot TLS | Potwierdzony status TLS/mTLS, brak ostrzeżeń o kluczach |
| 3. Wykonaj smoke test strategii (`python scripts/smoke_demo_strategies.py --cycles 1`) **lub** potwierdź ostatni przebieg workflow `Stage4 multi-strategy smoke` (zob. `docs/runbooks/STAGE4_MULTI_STRATEGY_SMOKE.md`) | Zespół Strategii | `reports/demo_smoke/*.json`, logi, `var/audit/acceptance/<TS>/stage4_smoke/*` | Wynik PASS, numer joba zapisany w runbooku Stage4, brak odchyleń > tolerancji |
| 4. Dodaj wpis do decision logu (`python scripts/verify_decision_log.py summary --append audit/decision_logs/demo.jsonl --stage demo --status demo_ready --summary-json var/audit/demo_summary.json --artefact config_sha384=<SHA-384> --tag demo`) | Operator Demo | `audit/decision_logs/demo.jsonl`, `var/audit/demo_summary.json` | Wpis podpisany HMAC (`signature.algorithm=HMAC-SHA256`), zawiera artefakt `summary_sha256` i `config_sha384`, tag `demo` oraz status `demo_ready` |

## Etap 2 – Promocja do paper
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Uruchom Paper Labs (`python scripts/run_risk_simulation_lab.py --profile all`) | Zespół Ryzyka | `reports/paper_labs/*.json`, `reports/paper_labs/*.pdf` | Wszystkie profile PASS, podpis Compliance |
| 2. Zweryfikuj bundla OEM (`python deploy/packaging/build_core_bundle.py --dry-run --platform linux`, komenda używa wbudowanych próbek z `deploy/packaging/samples/` lub generuje tymczasowe placeholdery, jeśli katalog próbek nie jest dostępny) | Release | Raport bundlera, `manifest.json` | Manifest podpisany, brak rozbieżności |
| 3. Provision licencję paper (`python scripts/oem_provision_license.py request.json --registry var/licenses/registry.jsonl`) | Operator OEM | `var/licenses/registry.jsonl`, licencja `.jsonl` | Licencja podpisana, wpis w decision logu |
| 4. Uruchom `python scripts/run_paper_smoke_ci.py --render-summary-markdown` | Zespół Runtime | `reports/paper_smoke/*.json`, `reports/paper_smoke/*.md` | Smoke PASS, raport przesłany do compliance |
| 5. Potwierdź checklistę `docs/runbooks/paper_trading.md` | Operator Paper | Wypełniona lista kontrolna | Sekcja „Akceptacja” podpisana przez Compliance |
| 6. Dodaj wpis `status=paper_ready` do `audit/decision_logs/paper.jsonl` | Compliance | Decision log paper | Wpis podpisany HMAC-SHA384, zweryfikowany `verify_decision_log.py` |

## Etap 3 – Promocja do live
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Wygeneruj pakiet mTLS (`python scripts/generate_mtls_bundle.py --output certs/live`) | Security | `certs/live/*`, rejestr rotacji | Komplet materiałów TLS, potwierdzony fingerprint CA |
| 2. Konfiguruj `config/core.yaml` sekcję `execution.live` oraz `alerts.prometheus` | Inżynier Runtime | Nowa wersja `config/core.yaml`, diff Git | PR zatwierdzony, hash wpisany do decision logu |
| 3. Uruchom `python scripts/live_execution_dry_run.py --config config/core.yaml --audit-json audit/decision_logs/live_execution.jsonl --dry-run` | Zespół Egzekucji | Raport dry-run, decision log | Dry-run PASS, brak błędów adapterów |
| 4. Przeprowadź checklistę `docs/runbooks/LIVE_EXECUTION_CHECKLIST.md` | Operator Live | Wypełniony formularz checklisty | Wszystkie pola `Akceptacja` oznaczone jako `[x]` |
| 5. Potwierdź gotowość UI (telemetria guard, reduce motion, fps monitor) | Zespół UI | Logi UI (`logs/ui_telemetry_alerts.jsonl`) | Guard aktywny, brak ostrzeżeń |
| 6. Wpisz `status=live_ready` w decision logu (`audit/decision_logs/live_execution.jsonl`) | Compliance | Decision log live | Wpis podpisany HMAC, zweryfikowany `verify_decision_log.py --strict` |

## Sekcja Stage6 – migracja profilu i sekretów
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Uruchom migrację Stage6 (`python -m bot_core.runtime.stage6_preset_cli --core-config config/core.yaml --preset presets/gui.json --profile-name stage6_gui --core-backup config/core.yaml.bak --core-diff --secrets-input secrets/preset.yaml --secrets-output secrets/api_keys.vault --secret-passphrase-file secrets/pass.txt --summary-json var/audit/stage6/migration_summary.json`) | Operator Stage6 | `config/core.yaml`, `config/core.yaml.bak`, `secrets/api_keys.vault`, `var/audit/stage6/migration_summary.json` | Diff z `--core-diff` przejrzany, kopia zapasowa zarchiwizowana, magazyn sekretów zaszyfrowany, sumy SHA-256 w `migration_summary.json` potwierdzone (core, backup, magazyn, źródłowe pliki sekretów), zarejestrowane źródła haseł (`secrets.output_passphrase` oraz – jeśli rotacja była wykonana – `secrets.rotation_passphrase`, każde z identyfikacją inline/plik/env) oraz brak ostrzeżeń w polu `warnings` (ewentualne wpisy udokumentowane w decision logu); sekcja `cli_invocation` zawiera zanonimizowaną listę argumentów (hasła zastąpione `***REDACTED***`), a sekcja `tool` rejestruje interpreter, wersję pakietu i rewizję git migratora (wartości potwierdzone w decision logu) |
| 2. Jeśli klient korzysta z zaszyfrowanego pliku `SecurityManager`, poinformuj go o konieczności użycia narzędzia migracyjnego opisanego w [docs/migrations/2024-legacy-storage-removal.md](../migrations/2024-legacy-storage-removal.md); runtime Stage6 nie obsługuje już odczytu tych plików. | Operator Stage6 | Notatka migracyjna | Potwierdzenie, że migracja została wykonana poza środowiskiem runtime (w decision logu) |
| 2. Dołącz plik `migration_summary.json` do decision logu (`audit/decision_logs/stage6.jsonl`) i podpisz wpis HMAC | Compliance/Risk Stage6 | Decision log Stage6, plik podsumowania | Wpis zawiera hash SHA-384 podsumowania oraz status `stage6_profile_ready` |

> **Podpowiedź CLI:** wywołanie `python -m bot_core.runtime.stage6_preset_cli --core-config config/core.yaml --preset presets/gui.json --help`
> prezentuje wyłącznie aktywne flagi migratora. Najważniejsze przełączniki:
> ```bash
> --core-config CORE_CONFIG      Ścieżka do docelowego pliku core.yaml
> --preset PRESET                Preset GUI (JSON/YAML) do zaimportowania
> --secrets-input PATH           Plik z danymi sekretów do zaczytania
> --secrets-output PATH          Docelowy magazyn EncryptedFileSecretStorage
> --secret-passphrase{,-env,-file}
>                                Dostarczenie hasła magazynu sekretów
> --summary-json PATH            Raport audytowy migracji (JSON)
>
> Nowe wdrożenia powinny korzystać wyłącznie z tych flag – migrator nie udostępnia już przełączników z poprzedniej warstwy.
> ```

## Artefakty/Akceptacja
- Decision logi: `audit/decision_logs/demo.jsonl`, `audit/decision_logs/paper.jsonl`, `audit/decision_logs/live_execution.jsonl` z podpisami HMAC-SHA384.
- Raporty Paper Labs (JSON + PDF), smoke paper, logi dry-run live.
- Hash SHA-384 konfiguracji `config/core.yaml` oraz bundla OEM przypisany do release.
- Checklisty podpisane przez odpowiedzialne role i zarchiwizowane w `var/audit/`, w tym `var/audit/stage6/migration_summary.json` z migracji Stage6 zawierający sumy SHA-256 dla `core.yaml`, kopii zapasowej, magazynu sekretów oraz źródłowych artefaktów (plików sekretów/`SecurityManager` + soli), identyfikację źródeł haseł (`secrets.output_passphrase` i – jeżeli rotacja była wykonywana – `secrets.rotation_passphrase`), potwierdzenie, że pole `warnings` jest puste (lub opisano zarejestrowane ostrzeżenia w decision logu), zanonimizowaną sekcję `cli_invocation` z argumentami CLI (hasła → `***REDACTED***`) oraz metadane sekcji `tool` (ścieżka interpretera, wersja Pythona, dostępność pakietu `dudzian-bot`, rewizja git).
