# Phase 2 – OEM: Plan wdrożenia

## Cel iteracji
- Dostarczyć w pełni offline'owy pakiet instalacyjny (daemon + UI Qt/QML) z podpisami kryptograficznymi.
- Uruchomić pipeline paper→live z rozszerzonym silnikiem ryzyka (profile + stres testy) oraz checklistami compliance.
- Zintegrować warstwę live execution z UI i alertami przy zachowaniu wymogów bezpieczeństwa (mTLS, RBAC, audyty).

## Zakres prac
### 1. Packaging OEM
- [x] Opracować format bundla "Core OEM" zawierający:
  - binaria demona (`bot_core`) oraz UI (`ui/` Qt/QML) w wersjach dla Linux/macOS/Windows,
  - prekonfigurowane pliki `.env` i `config/core.yaml` z podpisami HMAC (SHA-384 + klucze rotacyjne),
  - bootstrap skrypt instalacyjny z weryfikacją fingerprintu urządzenia (`deploy/packaging/bootstrap/*`).
- [x] Przygotować pipeline buildów:
  - GitHub Actions self-hosted (air-gapped) z jobami dla platform docelowych (`deploy/ci/github_actions_oem_packaging.yml`),
  - artefakty MSI (Windows), pkg/notarized dmg (macOS), AppImage + .deb (Linux) – publikowane jako artefakty CI (placeholder na binaria produkcyjne).
- [x] Wprowadzić proces podpisywania binariów:
  - Windows: SignTool + certyfikat EV/organizacyjny (konfiguracja w pipeline – krok podpisu do uzupełnienia kluczem produkcyjnym),
  - macOS: `codesign` + plan notarization (placeholder w jobie macOS, integracja z narzędziem Apple Notary w następnym kroku),
  - Linux: GPG + manifest SHA-384 (manifest generowany automatycznie, publikacja podpisu w `manifest.sig`).
- [x] Udokumentować procedurę "Installer Acceptance" (lista kontrolna artefakty/akceptacja) w `docs/runbooks/OEM_INSTALLER_ACCEPTANCE.md`.
- [x] Zautomatyzować dry-run akceptacyjny (`scripts/run_oem_acceptance.py`) generujący raport JSON z wynikami bundla/licencji/Paper Labs/mTLS.

### 2. Fingerprint i licencjonowanie offline
- [x] Zaimplementować moduł `bot_core/security/fingerprint.py` generujący fingerprint (CPU, TPM, MAC, opcjonalny dongle) z podpisem HMAC.
- [x] Zaprojektować rejestr licencji (`var/licenses/registry.jsonl`) z podpisami oraz narzędziem `scripts/oem_provision_license.py`.
- [x] Stworzyć runbook provisioning (checklista + artefakty: request, approval, signed license).
- [x] Uzupełnić UI o ekran aktywacji licencji z obsługą QR/USB.

### 3. Rozszerzenie silnika ryzyka
- [x] Dodać moduł symulacji scenariuszy (`bot_core/risk/simulation.py`) z obsługą profili: konserwatywny, zbalansowany, agresywny, manualny.
- [x] Przygotować stres testy (flash crash, dry liquidity, latency spike) bazujące na danych Parquet.
- [x] Zintegrować raporty PDF/JSON z pipeline'em gatingowym (`deploy/ci/github_actions_paper_smoke.yml`).
- [x] Przygotować runbook "Paper Labs" (lista kontrolna: dane wejściowe, wyniki symulacji, akceptacja compliance).

### 4. Live execution + UI + Alerting
- [x] Uzupełnić `bot_core/execution/live_router.py` o realne routowanie i fallbacki giełdowe.
- [x] Dodać monitoring latencji/fill rate (metryki Prometheus, alerty w `config/core.yaml`).
- [x] Rozszerzyć gRPC o mTLS + RBAC tokeny, generatory certyfikatów (`scripts/generate_mtls_bundle.py`).
- [x] Zaktualizować UI (gRPC client) o obsługę TLS, telemetry overlay, checklistę przed startem live.

### 5. Dokumentacja i runbooki
- [x] Checklisty demo→paper→live w formacie: Krok, Odpowiedzialny, Artefakty, Akceptacja.
- [x] Plan reinstalacji w środowisku offline (backup licencji, recovery bundli, walidacja hashy).
- [x] Decision log JSONL z podpisami (narzędzie `scripts/verify_decision_log.py`).

## Wymagania bezpieczeństwa
- mTLS (TLS 1.3, certyfikaty rotowane co 90 dni, CRL offline).
- RBAC oparty na tokenach z podpisem HMAC (audyt JSONL).
- Brak WebSocketów – wyłącznie gRPC/HTTP2 lub IPC.
- Guard UI: FPS monitor, reduce motion, p95 <150 ms, jank <1%.

> **Blokady architektoniczne:** Przejście do środowiska live jest sterowane mechanizmami `StrategyContext.require_demo_mode`, flagami `compliance_confirmed` oraz walidacją konfiguracji opisanymi w `docs/ARCHITECTURE.md`. Milestony sprintów 3–4 wymagają kompletnego zestawu decision logów (`demo.jsonl`, `paper.jsonl`, `live_execution.jsonl`) oraz podpisanych checklist zanim runtime odblokuje tryb live.

## Harmonogram (propozycja)
| Sprint | Zakres | Kluczowe zadania | Milestone / Artefakty gatingowe |
| --- | --- | --- | --- |
| 1 | Packaging OEM + podpisy + weryfikacja demo | Finalizacja bundla `Core OEM`, podpisy binariów, przygotowanie pipeline'u `deploy/ci/github_actions_oem_packaging.yml`, uruchomienie kroków Etapu 1 runbooka Demo→Paper (walidacja `config/core.yaml`, smoke demo, wpis `status=demo_ready`). | Artefakty instalatora + checklisty instalacyjne podpisane w `docs/runbooks/OEM_INSTALLER_ACCEPTANCE.md`, hash `config/core.yaml` (SHA-256/384) zapisany w `audit/decision_logs/demo.jsonl`, `StrategyContext.require_demo_mode` utrzymany jako blokada. |
| 2 | Fingerprint/licencje + provisioning | Ukończony moduł fingerprintu, rejestr licencji (`scripts/oem_provision_license.py`), zadania runbookowe Stage "Provision licencji" oraz Etapu 2 krok 3 (provision licencji paper). | Pierwszy klient OEM offline z licencją podpisaną HMAC-SHA384, wpis w `audit/decision_logs/demo.jsonl` zawierający hash `config/core.yaml` oraz potwierdzenie `runtime.compliance_confirmed=false` do czasu ukończenia Paper Labs. |
| 3 | Symulacje/stres testy + gating paper | Dedykowane zadania runbooku Demo→Paper: Paper Labs, walidacja bundla OEM (dry-run CI), provisioning licencji paper, smoke paper (`deploy/ci/github_actions_paper_smoke.yml`), podpis checklisty `docs/runbooks/paper_trading.md`. | Raporty "Paper Labs" + smoke paper przesłane do compliance, `status=paper_ready` w `audit/decision_logs/paper.jsonl`, `StrategyContext.require_demo_mode` zdejmuje blokadę paper dopiero po `verify_decision_log.py --strict`. |
| 4 | Live execution + UI TLS | Checklisty live (UI telemetry, mTLS bundle, dry-run live) z `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md`, aktualizacja `config/core.yaml` (hash) i rotacja certyfikatów. | Tryb live pilotowy aktywowany po `status=live_ready` w `audit/decision_logs/live_execution.jsonl`, flagach `runtime.compliance_confirmed=true` oraz zatwierdzonych checklistach paper/live; `StrategyContext.require_demo_mode` i walidacja `config/core.yaml` w runtime blokują start do czasu spełnienia warunków z `docs/ARCHITECTURE.md`. |

### Zadania sprintowe z runbooka Demo → Paper → Live

#### Sprint 1 – Hardening demo i bundla OEM
- [ ] Zweryfikować `config/core.yaml` poprzez `python scripts/validate_config.py --profile demo` i zapisać hash (SHA-256/SHA-384) w `audit/decision_logs/demo.jsonl` wraz z numerem commita bundla OEM.
- [ ] Uruchomić smoke demo (`python scripts/smoke_demo_strategies.py --cycles 1`) lub potwierdzić najnowszy run Stage4 (patrz `docs/runbooks/STAGE4_MULTI_STRATEGY_SMOKE.md`) i zarchiwizować raport w `var/audit/acceptance/`.
- [ ] Wykonać test `python scripts/run_metrics_service.py --shutdown-after 0 --jsonl logs/metrics_demo.jsonl` i dołączyć wynik do checklisty Etapu 1 (`docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md`).
- [ ] Dodać wpis `status=demo_ready` w `audit/decision_logs/demo.jsonl` podpisany HMAC przy pomocy `python scripts/verify_decision_log.py --strict`.

#### Sprint 2 – Przygotowanie provisioning offline
- [ ] Wykonać krok "Provision licencji paper" z `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md` (skrypt `scripts/oem_provision_license.py`, rejestr `var/licenses/registry.jsonl`, podpis HMAC).<br>
  **Artefakty:** wpis decyzji `status=demo_ready` oraz załączony hash `config/core.yaml` (SHA-256 z `python scripts/validate_config.py`).
- [ ] Zarejestrować w decision logu demowym `runtime_flags` (`require_demo_mode=true`, `compliance_confirmed=false`) poprzez `python scripts/verify_decision_log.py update --log audit/decision_logs/demo.jsonl`.

#### Sprint 3 – Gating paper (Paper Labs + smoke)
- [ ] Uruchomić Paper Labs `python scripts/run_risk_simulation_lab.py --profile all` i załączyć raporty JSON/PDF (`reports/paper_labs/*`).<br>
  **Referencja:** sekcja "Etap 2 – Promocja do paper" w `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md`.
- [ ] Zweryfikować bundla OEM zgodnie z krokiem "Zweryfikuj bundla OEM" (dry-run `deploy/packaging/build_core_bundle.py`) i opublikować manifest CI (`deploy/ci/github_actions_oem_packaging.yml`).
- [ ] Wykonać provisioning licencji paper (oddzielnie od sprintu 2 dla środowiska paper) oraz zarejestrować podpis w `audit/decision_logs/paper.jsonl`.
- [ ] Uruchomić smoke paper `python scripts/run_paper_smoke_ci.py --render-summary-markdown` lub job `deploy/ci/github_actions_paper_smoke.yml` i dołączyć raport Markdown do review compliance.
- [ ] Potwierdzić checklistę `docs/runbooks/paper_trading.md` (sekcja akceptacji podpisana) i załączyć numer joba CI do decision logu.
- [ ] Zaktualizować `audit/decision_logs/paper.jsonl` wpisem `status=paper_ready`, hashami artefaktów (`reports/paper_labs/*.pdf`, `reports/paper_smoke/*.md`, `config/core.yaml`) oraz flagą `runtime.compliance_confirmed=false` (blokada przejścia live) potwierdzoną `python scripts/verify_decision_log.py --strict`.

#### Sprint 4 – Blokady przed live
- [ ] Wykonać kroki etapu live z `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md`: generowanie pakietu mTLS (`scripts/generate_mtls_bundle.py`), aktualizacja `config/core.yaml` (hash w decision logu), dry-run live (`scripts/live_execution_dry_run.py`).
- [ ] Podpisać checklistę `docs/runbooks/LIVE_EXECUTION_CHECKLIST.md` i dołączyć log telemetryczny UI (`logs/ui_telemetry_alerts.jsonl`).
- [ ] Potwierdzić gotowość UI (telemetria guard) – referencja do `deploy/ci/ui_telemetry_checks.yml` lub manualny raport z `UiTelemetryAlertSink`.
- [ ] Wpisać `status=live_ready` w `audit/decision_logs/live_execution.jsonl` po pozytywnym wyniku dry-run i podpisach compliance (wymóg `docs/ARCHITECTURE.md`).
- [ ] Złożyć w decision logu live podpisany wpis z metadanymi (`config/core.yaml` hash SHA-384, identyfikator joba `deploy/ci/github_actions_paper_smoke.yml`, fingerprint paczki mTLS) oraz potwierdzeniem ustawienia `StrategyContext.require_demo_mode=false` i `runtime.compliance_confirmed=true` dopiero po `verify_decision_log.py --strict`.

### Bramki decyzyjne Demo → Paper → Live

| Bramka | Minimalny zestaw artefaktów | Runbook / CI | Walidacje dodatkowe |
| --- | --- | --- | --- |
| **Demo exit** | \- Hash `config/core.yaml` (SHA-256/SHA-384) + podpis HMAC<br>\- Raport smoke demo (`var/audit/acceptance/*.jsonl`)<br>\- Checklisty Etapu 1 (`docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md`) podpisane przez inżyniera + właściciela produktu | \- `scripts/smoke_demo_strategies.py`<br>\- `deploy/ci/github_actions_oem_packaging.yml` (art. bundla)<br>\- `docs/runbooks/OEM_INSTALLER_ACCEPTANCE.md` | \- `StrategyContext.require_demo_mode=true`<br>\- `verify_decision_log.py --strict --log audit/decision_logs/demo.jsonl` |
| **Paper exit** | \- Raport Paper Labs (`reports/paper_labs/*.pdf` + `reports/paper_labs/*.json`)<br>\- Wynik smoke paper (`reports/paper_smoke/*.md`)<br>\- Wpisy `status=paper_ready` + `runtime.compliance_confirmed=false` | \- `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md` – Etap 2<br>\- `docs/runbooks/paper_trading.md`<br>\- `deploy/ci/github_actions_paper_smoke.yml` | \- Hash `config/core.yaml` i fingerprint bundla z `deploy/packaging/build_core_bundle.py`<br>\- Decyzja compliance (HMAC) w `audit/decision_logs/paper.jsonl` |
| **Live go** | \- Podpisane checklisty live (`docs/runbooks/LIVE_EXECUTION_CHECKLIST.md`)<br>\- Raport dry-run (`logs/live_dry_run/*.jsonl`)<br>\- Pakiet mTLS (`var/certs/live_bundle.tar.gz` + fingerprint) | \- `scripts/live_execution_dry_run.py`<br>\- `scripts/generate_mtls_bundle.py`<br>\- `deploy/ci/ui_telemetry_checks.yml` | \- `StrategyContext.require_demo_mode=false` potwierdzone w `audit/decision_logs/live_execution.jsonl`<br>\- Walidacja hashy `config/core.yaml` (paper vs live) + kontrola `runtime.compliance_confirmed=true` |

**Notatka:** Każda bramka wymaga dołączenia referencji do numerów runów CI (job id + commit) oraz zarchiwizowania artefaktów w `audit/artefacts/<data>/<gate>/`. Sekcja "Wymagania bezpieczeństwa" w `docs/ARCHITECTURE.md` określa, że zmiana flag `StrategyContext` jest dozwolona wyłącznie po pozytywnej weryfikacji `verify_decision_log.py --strict` i potwierdzeniu podpisem HMAC właściciela systemu.

### Decision log – schemat wpisu

Każdy wpis w `audit/decision_logs/*.jsonl` powinien posiadać strukturę:

```json
{
  "timestamp": "2025-08-18T14:25:04Z",
  "stage": "paper",
  "status": "paper_ready",
  "artefacts": {
    "config_hash": "sha384:...",
    "paper_labs_report": "reports/paper_labs/2025-08-18-labs.pdf",
    "smoke_run_id": "github_actions_paper_smoke.yml#12345",
    "mtls_fingerprint": null
  },
  "runtime_flags": {
    "StrategyContext.require_demo_mode": false,
    "runtime.compliance_confirmed": false
  },
  "signatures": {
    "owner": "hmac:...",
    "compliance": "hmac:..."
  }
}
```

Schemat jest walidowany przez `python scripts/verify_decision_log.py --strict --schema docs/schemas/decision_log_v2.json`, a brak dowodu (`artefacts.*`) blokuje zatwierdzenie kolejnego etapu. Przy przejściu do live dodatkowo wymagamy dołączenia hashy `var/certs/live_bundle.tar.gz` oraz fingerprintów urządzeń licencjonowanych (`var/licenses/registry.jsonl`).

## Ryzyka i mitigacje
| Ryzyko | Mitigacja |
| --- | --- |
| Brak certyfikatów EV/Apple | Zamówienie certyfikatów równolegle, fallback self-signed + whitelist lokalna |
| Niespójne fingerprinty HW | Warstwa abstrakcji + fallback dongle/OTP |
| Opóźnienia w integracji UI | Równoległy prototyp CLI + testy gRPC contract |
| Brak danych do stres testów | Generatory syntetyczne + import publicznych datasetów |

## Artefakty wyjściowe
- Instalatory OEM (MSI, pkg/dmg, AppImage/.deb) z podpisami, manifestem i raportami z pipeline'u `deploy/ci/github_actions_oem_packaging.yml`.
- Repozytorium licencji offline + narzędzia provisioning wraz z podpisanymi wpisami `audit/decision_logs/*.jsonl` (`demo_ready`, `paper_ready`, `live_ready`).
- Raporty Paper Labs (`reports/paper_labs/*.json`, `*.pdf`), smoke paper (`reports/paper_smoke/*.json`, `*.md`) oraz checklisty `docs/runbooks/paper_trading.md` i `docs/runbooks/LIVE_EXECUTION_CHECKLIST.md` podpisane przez odpowiedzialne role.
- Hash i podpis HMAC konfiguracji `config/core.yaml` (SHA-256/SHA-384) zarejestrowany w decision logach oraz weryfikowany `python scripts/validate_config.py` i `scripts/verify_decision_log.py`.
- Zaktualizowana dokumentacja runbooków (`docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md`) oraz logi telemetryczne UI (`logs/ui_telemetry_alerts.jsonl`) potwierdzające guardy TLS/RBAC przed startem live.

## Referencje operacyjne
- Runbook Demo → Paper → Live: `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md` – obowiązkowe kroki gatingowe dla sprintów 2–4.
- Runbook Paper trading: `docs/runbooks/paper_trading.md` – potwierdza kompletność środowiska paper wraz z raportem smoke CI (`deploy/ci/github_actions_paper_smoke.yml`).
- Runbook Live execution: `docs/runbooks/LIVE_EXECUTION_CHECKLIST.md` – wymagany podpis przed aktywacją live, w zestawie z raportem `scripts/live_execution_dry_run.py`.
- Decision logi (`audit/decision_logs/*.jsonl`) weryfikowane przez `scripts/verify_decision_log.py --strict` – jedyny mechanizm dopuszczający przejście środowiska zgodnie z blokadami z `docs/ARCHITECTURE.md`.

