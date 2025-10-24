# Stage6 – Checklista przygotowawcza

## 1. Zakres
Lista kontrolna służy do potwierdzenia gotowości środowiska i zespołu przed rozpoczęciem
implementacji Etapu 6. Zawiera zweryfikowane wymagania operacyjne, zależności konfiguracyjne
oraz punkty kontrolne dla skryptów `scripts/run_stage6_*.py` i szablonów w `config/stage6/`.

## 2. Kontrola danych
- [ ] Zidentyfikowano wymagane źródła depth-of-book / sentiment i potwierdzono możliwość
      pozyskania offline (`data/stage6/*`).
- [ ] Przygotowano plan rozszerzeń manifestów Parquet/SQLite dla danych Stage6 (aktualizacja
      manifestów Market Intel i Stress Lab).
- [ ] Zweryfikowano zapotrzebowanie na dodatkowe zasoby storage / compute dla pipeline'ów
      Market Intel i Stress Lab.
- [ ] Potwierdzono możliwość generowania raportów Market Intel (`python
      scripts/build_market_intel_metrics.py`) do lokalizacji wskazanych w
      `config/stage6/hypercare.yaml` i `scripts/run_stage6_portfolio_cycle.py`.
- [ ] Zapewniono katalog `var/metrics/` z metrykami Stage6 (`stage6_measurements.json`) –
      jeśli plik pochodzi z innego repozytorium, użyj
      `python scripts/sync_stage6_metrics.py --source <plik> --output var/metrics/stage6_measurements.json`,
      który utworzy katalog docelowy, zweryfikuje poprawność JSON oraz liczbę pomiarów
      i przygotuje plik do użycia w orchestracji hypercare, oraz retencję artefaktów
      w `var/audit/stage6/`.

## 3. PortfolioGovernor & Decision Engine
- [ ] Zdefiniowano KPI adaptacyjnego zarządzania kapitałem (np. target allocation drift, max
      drawdown) i powiązano je z raportem generowanym przez
      `python scripts/run_stage6_portfolio_cycle.py`.
- [ ] Określono interfejs integracji PortfolioGovernora z DecisionOrchestrator i schedulerem
      multi-strategy (`bot_core.runtime`).
- [ ] Ustalono procedury override operatora oraz wymagane wpisy decision logu
      (`resolve_decision_log_config`, `default_decision_log_path`).
- [ ] Zweryfikowano ścieżki wejściowe portfela (alokacje, Market Intel, raporty SLO/Stress Lab)
      zgodnie z sekcją `portfolio` w `config/stage6/hypercare.yaml`.
- [ ] Potwierdzono, że wartości `inputs.*` i `output.*` w `config/stage6/hypercare.yaml`
      odpowiadają parametrom CLI (`--allocations`, `--market-intel`, `--portfolio-value`,
      `--summary`, `--summary-signature`, `--summary-csv`) w `scripts/run_stage6_portfolio_cycle.py`
      oraz domyślnym ścieżkom (`var/audit/portfolio/`).
- [ ] Zweryfikowano konfigurację katalogów awaryjnych i wymaganych symboli (`inputs.fallback_dirs`,
      `inputs.market_intel_required_symbols`, `inputs.*_max_age`) względem flag CLI
      `--fallback-dir`, `--market-intel-required`, `--market-intel-max-age`, `--slo-max-age`,
      `--stress-max-age`.
- [ ] Przygotowano klucze HMAC do podpisów raportów portfelowych (`--signing-key`,
      `--signing-key-path`, `--signing-key-id`).
- [ ] Zaplanowano rejestrowanie metadanych (`--metadata`, `--log-context`) w raportach
      `PortfolioHypercareCycle`.

## 4. Observability & Alerting
- [ ] Potwierdzono aktualność definicji SLO (`config/observability/slo.yml`) i mapowań severity
      wymaganych przez `python scripts/run_stage6_observability_cycle.py`.
- [ ] Zapewniono źródła metryk (`--metrics`, `--bundle-source`) oraz katalog `var/observability/`
      na paczki obserwowalności.
- [ ] Skonfigurowano generowanie override'ów alertów (`--overrides-json`, `--overrides-ttl`,
      `--tag stage6`) oraz podpisów HMAC.
- [ ] Zweryfikowano ścieżki i identyfikatory kluczy HMAC (`--signing-key`, `--signing-key-env`,
      `--signing-key-path`, `--signing-key-id`) z sekcją `observability.signing` w
      `config/stage6/hypercare.yaml`.
- [ ] Potwierdzono spójność parametrów TTL, źródła i tagów override'ów (`observability.overrides`
      → `ttl_minutes`, `requested_by`, `source`, `include_warning`, `tags`,
      `severity_overrides`) z flagami CLI `--overrides-ttl`, `--overrides-requested-by`,
      `--overrides-source`, `--skip-warning`, `--tag`, `--severity` oraz planem scalania
      istniejących wpisów (`--existing-overrides`).
- [ ] Udokumentowano synchronizację dashboardów Grafana (`--dashboard`, `--annotations-output`,
      `--annotations-signature`).
- [ ] Zweryfikowano podpisy raportów SLO i override w `var/audit/observability/` oraz paczki
      obserwowalności w `var/observability/`, w tym opcję weryfikacji pakietu
      (`--no-verify-bundle`).
- [ ] Potwierdzono, że `config/stage6/hypercare.yaml` → `observability.bundle` wskazuje katalog
      `var/observability`, nazwę `stage6-observability` i źródła (`deploy/grafana/...`,
      `deploy/prometheus`) zgodne z parametrami `--bundle-output-dir`, `--bundle-name` i
      `--bundle-source` w `scripts/run_stage6_observability_cycle.py`.
- [ ] Uzgodniono świadome użycie wzorców `bundle.include` / `bundle.exclude` oraz metadanych bundla
      (`--bundle-include`, `--bundle-exclude`, `--bundle-metadata`) – potwierdzono, że konfiguracja
      Stage6 odzwierciedla wymagane profile (`observability.bundle.metadata`).
- [ ] Przygotowano listę metadanych bundla (profile, źródła) zgodnie z przykładem z
      `config/stage6/hypercare.yaml`.

## 5. Stress Labs & Resilience
- [ ] Zebrano przypadki testów DR/stres (blackout infrastrukturalny, degradacja giełdy,
      awaria adaptera) i odnotowano je w `STAGE6_STRESS_LAB_CHECKLIST.md` oraz
      `STAGE6_RESILIENCE_DRILL_CHECKLIST.md`.
- [ ] Zaplanowano harmonogram ćwiczeń `python scripts/failover_drill.py` i wymagane artefakty
      audytowe (`var/audit/resilience/`).
- [ ] Przygotowano wymagania SLO2 i progi alertów dla obserwowalności Stage6 powiązane z
      raportami Stress Lab.
- [ ] Zweryfikowano konfiguracje bundlowania i audytu resilience (`python
      scripts/run_stage6_resilience_cycle.py`, `python scripts/export_resilience_bundle.py`,
      `python scripts/verify_resilience_bundle.py`).
- [ ] Potwierdzono ścieżki paczek i podpisów (`bundle`, `audit`, `failover`, `self_healing`)
      zgodnie z `config/stage6/hypercare.yaml` i szablonem
      `config/stage6/resilience_self_heal.json`.
- [ ] Sprawdzono, że `resilience.bundle.*`, `resilience.audit.*`, `resilience.failover.*` oraz
      `resilience.self_healing.*` w `config/stage6/hypercare.yaml` są zgodne z domyślnymi
      parametrami `scripts/run_stage6_resilience_cycle.py` (np. `--bundle-output-dir`,
      `--bundle-name`, `--audit-json`, `--failover-json`, `--self-heal-output`).
- [ ] Zweryfikowano odwzorowanie wzorców `resilience.bundle.include/exclude` i metadanych na
      parametry CLI (`--include`, `--exclude`, `--metadata`) oraz plan podpisów HMAC dla każdego
      artefaktu bundla.
- [ ] Zweryfikowano konfigurację kluczy HMAC (`--signing-key*`, `--audit-hmac-key*`) z sekcjami
      `resilience.signing` i `resilience.audit_hmac` w `config/stage6/hypercare.yaml`.
- [ ] Potwierdzono politykę audytu (`resilience.audit.require_signature`, `resilience.audit_hmac`)
      z flagami `--audit-require-signature`, `--audit-no-verify`, `--audit-policy` i
      `--audit-hmac-key*`, w tym scenariusze, w których weryfikacja podpisu manifestu jest
      celowo pomijana.
- [ ] Udokumentowano procedury podpisów HMAC i weryfikacji (`--signing-key-path`,
      `--audit-hmac-key`, `--audit-hmac-key-path`, `--audit-hmac-key-env`) dla raportów resilience.

## 6. Operacje i compliance
- [ ] Ustalono skład zespołu warsztatowego Stage6 (trading, risk, compliance, operations L1/L2)
      i przypisano odpowiedzialności za sekcje runbooków Stage6.
- [ ] Zarejestrowano warsztaty discovery w decision logu (`python scripts/log_stage5_training.py`)
      z podpisem HMAC.
- [ ] Przygotowano repozytorium artefaktów `var/audit/stage6_discovery/` (podpisy HMAC, struktura
      katalogów, retencja 24+ m-cy).
- [ ] Zapewniono politykę rotacji kluczy HMAC dla Stage6 (observability, resilience, portfolio,
      runbooki) oraz repozytorium kluczy w `secrets/hmac/`.
- [ ] Zaktualizowano runbook demo → paper → live o sekcję Stage6
      (`DEMO_PAPER_LIVE_CHECKLIST.md`) z kontrolą migracji presetów.
- [ ] Zapewniono proces review/akceptacji runbooków Stage6 przez interesariuszy (product,
      compliance, operations).
- [ ] Potwierdzono utrzymanie repozytorium kluczy HMAC (`secrets/hmac/stage6_runbooks.key`,
      `secrets/hmac/observability.key`, `secrets/hmac/resilience*.key`,
      `secrets/hmac/stage6_portfolio.key`) oraz aktualność polityk rotacji w
      `config/stage6/hypercare.yaml` i powiązanych runbookach.

## 7. Automatyzacja Stage6 (Hypercare)
- [ ] Przygotowano i zrecenzowano konfigurację `config/stage6/hypercare.yaml` (sekcje `summary`,
      `observability`, `resilience`, `portfolio`) dla docelowych środowisk.
- [ ] Zweryfikowano uruchomienie `python scripts/run_stage6_hypercare_cycle.py --config
      config/stage6/hypercare.yaml` na danych testowych, w tym generowanie podpisanego
      `hypercare_summary.json` oraz `.sig`.
- [ ] Sprawdzono spójność ustawień podpisów HMAC w sekcjach `summary.signing`,
      `observability.signing`, `resilience.signing`, `resilience.audit_hmac` oraz
      `portfolio.signing` z kluczami w `secrets/hmac/` i parametrami CLI.
- [ ] Zweryfikowano, że konfiguracja hypercare przekazuje komplet parametrów wymaganych przez
      `Stage6HypercareCycle` (ścieżki raportów, bundli, decyzji) oraz że automatycznie propaguje
      ustawienia do poszczególnych skryptów (`run_stage6_observability_cycle.py`,
      `run_stage6_resilience_cycle.py`, `run_stage6_portfolio_cycle.py`).
- [ ] Skonfigurowano weryfikację zbiorczego raportu (`python
      scripts/verify_stage6_hypercare_summary.py --require-signature`).
- [ ] Zaplanowano integrację z pełną checklistą hypercare (`FULL_HYPERCARE_CHECKLIST.md`) oraz
      pipeline'em CI (połączenie Stage5/Stage6).
- [ ] Przygotowano wpisy decision logu dokumentujące wykonanie cyklu hypercare i podpisy HMAC.

## 8. Akceptacja wstępna
- [ ] Discovery Stage6 zatwierdzone przez product/compliance (wpis w decision logu + podpisy HMAC).
- [ ] Specyfikacja Stage6 (`docs/architecture/stage6_spec.md`) podpisana HMAC i zarchiwizowana w
      `var/audit/stage6/`.
- [ ] Runbook demo → paper → live zaktualizowany o sekcję Stage6 (migracja presetów i sekretów
      `preset_editor_cli --core-diff --core-backup --summary-json` z sumami kontrolnymi SHA-256 w
      podsumowaniu obejmującymi core, backup, magazyn oraz źródła sekretów i opcjonalną sól, wraz z
      identyfikacją źródeł haseł inline/plik/env oraz kontrolą, że pole `warnings` w
      `migration_summary.json` jest puste lub odnotowane w decision logu; sekcja `cli_invocation`
      przechowuje zanonimizowane argumenty CLI z hasłami zastąpionymi `***REDACTED***`, a sekcja
      `tool` rejestruje interpreter, wersję pakietu i rewizję git migratora i jest archiwizowana w
      decision logu).
- [ ] Zaktualizowana checklista wstępna podpisana HMAC i zarchiwizowana w `var/audit/stage6/` wraz
      z metadanymi review.

## 9. Artefakty i archiwizacja
- [ ] `docs/runbooks/STAGE6_PRELIMINARY_CHECKLIST.md` – wersja zatwierdzona + podpis HMAC
      (`var/audit/stage6/stage6_preliminary_checklist.md.sig`).
- [ ] Raporty discovery (`var/audit/stage6_discovery/<ts>/`), hypercare
      (`var/audit/stage6/hypercare_summary*.json/.sig`), resilience (`var/audit/resilience/*`),
      observability (`var/audit/observability/*`) i portfolio (`var/audit/portfolio/*`).
- [ ] Decision log Stage6 z wpisami dla discovery, migracji presetów, hypercare oraz warsztatów.
- [ ] Katalog `var/audit/stage6/` zawierający kopie runbooków, podpisy i pliki SHA-256/SHA-384
      zgodnie z wymaganiami compliance.
