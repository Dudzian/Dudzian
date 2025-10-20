# Warsztat operatorów – konfiguracja scheduler-a i presetów profili ryzyka

## Agenda (2h)
1. **Przegląd architektury** (15 min)
   - Moduły runtime: scheduler, strategie, silnik ryzyka, adaptery.
   - Ścieżka demo → paper → live i punkty kontrolne RBAC/HMAC.
2. **Konfiguracja `config/core.yaml`** (35 min)
   - Sekcja `runtime.multi_strategy_schedulers`: harmonogramy, `rbac_tokens`, `health_check_interval`.
   - Nowa sekcja `runtime.resource_limits`: limity CPU/RAM/I/O, próg ostrzeżeń.
   - Profile ryzyka (`risk_profiles`) – mapowanie `strategy_allocations`, `instrument_buckets`.
3. **Narzędzia operacyjne** (35 min)
   - `python scripts/audit_security_baseline.py --scheduler-required-scope runtime.schedule.write` – audyt RBAC/mTLS.
   - `python scripts/load_test_scheduler.py --iterations 30 --schedules 3` – szybki benchmark latencji i budżetów.
   - `python scripts/smoke_demo_strategies.py --cycles 3` – test dymny pipeline’u demo.
  - `python KryptoLowca/scripts/preset_editor_cli.py --core-config config/core.yaml --legacy-preset presets/default.json --profile-name legacy_gui --secrets-input secrets/gui.yaml --desktop-root KryptoLowca/ui/trading --secret-passphrase-file secrets/pass.txt` – migracja presetów GUI (JSON/YAML) do formatu Stage6 (`config/core.yaml`) oraz – jeśli podasz plik z sekretami – eksport do domyślnego magazynu (`api_keys.vault`) opartego o `EncryptedFileSecretStorage`. Dodanie `--dry-run` pozwala wydrukować docelowy YAML i pominąć zapis jakichkolwiek plików (sekrety również nie są wtedy eksportowane). Filtry `--secrets-include/--secrets-exclude` ograniczają migrację do konkretnych wpisów lub wzorców glob (np. `binance_*`, `*_token`), a CLI raportuje brakujące lub pominięte klucze i nie tworzy magazynu, gdy po filtrach nic nie zostanie. Włącz `--secrets-preview`, aby przed zapisem zobaczyć listę kluczy, które przejdą migrację (bez ujawniania wartości) – tryb podglądu działa również bez `--secrets-output`, co ułatwia szybki dry-run tylko z analizą kluczy. Skorzystaj z `--core-backup`, aby przed nadpisaniem `core.yaml` utworzyć kopię zapasową (`core.yaml.bak` lub ścieżka podana ręcznie), a `--core-diff` wydrukuje różnice względem poprzedniej zawartości, co pozwala szybko zweryfikować zmiany profili i ryzyka.
   - `python KryptoLowca/scripts/preset_editor_cli.py --core-config config/core.yaml --legacy-preset presets/default.json --profile-name legacy_gui --legacy-security-file secrets/api_keys.enc --legacy-security-passphrase-env LEGACY_SECURITY_PASS --secrets-output secrets/api_keys.vault --secret-passphrase stage6-pass` – migracja sekretów bezpośrednio z zaszyfrowanego pliku `SecurityManager` do nowego magazynu Stage6 (hasło można przekazać także przez plik `--legacy-security-passphrase-file`).
4. **Ćwiczenie praktyczne** (25 min)
   - Zadanie: dostroić preset `balanced` pod wyższy udział arbitrażu (20%).
   - Krok 1: Edycja `config/core.yaml` (`risk_profiles.balanced.strategy_allocations`).
   - Krok 2: `pytest tests/test_config_loader.py::test_load_core_config_resource_limits` – potwierdzenie poprawności.
   - Krok 3: `python scripts/load_test_scheduler.py --schedules 3 --signals 5 --output logs/workshop/balanced.json` – ocena latencji.
   - Krok 4: Dokumentacja zmian w decision logu (`verify_decision_log.py summary --append`).
5. **Q&A i zadania domowe** (10 min)
   - Przygotować własny preset profilu ryzyka (np. `balanced_high_vol`) i zarejestrować w runbooku.
   - Przećwiczyć rollback wg `docs/runbooks/rollback_multi_strategy.md`.

## Materiały
- `docs/architecture/stage4_spec.md`, `docs/architecture/stage4_test_plan.md`.
- `docs/runbooks/operations/strategy_incident_playbook.md` – procedury awaryjne.
- Logi przykładowe: `logs/load_tests/scheduler_profile.sample.json`, `audit/security/security_baseline.json`.

## Notatki dla prowadzącego
- Sprawdź dostępność kluczy RBAC (`CORE_SCHEDULER_TOKEN`) i datasetów w `data/backtests/normalized/`.
- Przygotuj konto uczestnika z uprawnieniami `runtime.schedule.write` (token `scheduler-local`).
- Monitoruj budżety zasobów podczas ćwiczenia: `python scripts/audit_security_baseline.py --print`.
