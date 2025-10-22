# Runbook: Paper → Live Transition

Proces przejścia z trybu paper trading do środowiska live wymaga potwierdzenia kontroli zgodnie z `docs/ARCHITECTURE.md` i checklistami bezpieczeństwa. Poniższa procedura prowadzi operatora przez wymagane kroki operacyjne oraz punkty eskalacji.

## 1. Pre-checklista

| Element | Weryfikacja | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| KYC/AML | Potwierdź podpisane pakiety compliance (`runtime_entrypoints.*.compliance`) i aktualność sign-offów. | `var/audit/`, raporty compliance, decyzje risk | Compliance + Security |
| Limity ryzyka | Sprawdź `risk_profiles` (max_daily_loss_pct, hard_drawdown_pct, max_position_pct, max_open_positions). | `config/core.yaml`, raporty RiskEngine | Risk |
| Alerty i audyt | Zweryfikuj kanały (`alert_channels`), throttling (`alert_throttle`) oraz backend audytu (FileAlertAuditLog). | Konfiguracja runtime, logi `alerts/` | SRE |
| Licencje | Potwierdź ważność licencji OEM i modułów live. | `var/licenses/active/`, log weryfikacji | Security |
| Dostęp do danych | Upewnij się, że źródła danych (OHLCV, risk snapshot) są kompletne dla rynku live. | Raporty data-quality, `var/data/<env>/` | Data |

## 2. Procedura aktywacji

1. **Freeze konfiguracji** – zatwierdź commit konfiguracyjny (core.yaml, sekcja runtime) i archiwizuj w `var/audit/stage6/` wraz z sumami SHA-256.
2. **Sekcja KYC/AML** – porównaj `signoffs` z aktualnym rejestrem compliance. Brakujące podpisy blokują migrację.
3. **Limity ryzyka** – porównaj konfigurację z raportami paper tradingu. Zaktualizuj `RiskProfile` jeśli dane papierowe wskazują rozbieżności.
4. **Alerty** – wykonaj test integracyjny alertów (wysyłka na kanały krytyczne, potwierdzenie audytu). Zweryfikuj brak throttlingu `__suppressed__` dla krytycznych kategorii.
5. **Secret Manager** – odśwież klucze `trade` (rotacja ≤90 dni) i zaktualizuj allowlist IP. Zanotuj w `security/rotation_log.json`.
6. **AutoTrader/GUI** – jeżeli `trusted_auto_confirm` ustawione na `true`, potwierdź w logach start bez manualnego kliknięcia. W trybie demo/paper pozostaw `false`.
7. **Dry-run** – uruchom pipeline w trybie read-only (`enable_auto_trade=False`) i sprawdź, czy `live_readiness_checklist` zgłasza status `ok` dla wszystkich elementów.
8. **Aktywacja** – odblokuj handel (ustaw `enable_auto_trade=True` lub usuń `require_demo_mode`). Monitoruj metryki w pierwszych 60 minutach.

## 3. Eskalacja i powrót do paper

- **Alert krytyczny / brak potwierdzenia** – natychmiast przełącz `enable_auto_trade` na `False`, włącz tryb awaryjny w RiskEngine (`risk_engine.activate_emergency_mode()`), eskaluj do #sec-alerts.
- **Niespełnione limity** – jeżeli `live_readiness_checklist` zgłosi status `blocked`, zatrzymaj bootstrap i wykonaj korekty konfiguracji przed kolejną próbą.
- **Awaria adaptera** – przy `failover_status.reconnect_required` > 0 uruchom runbook `docs/runbooks/STAGE6_RESILIENCE_CHECKLIST.md` i zdegraduj handel do paper/testnet.

## 4. Checklist zamknięcia okna

1. Archiwizuj logi audytu (`alerts/`, `risk_decisions.jsonl`, `live_execution`) w `var/audit/live/<YYYYMMDD>/`.
2. Uzupełnij raport `migration_summary.json` o status KYC, limity, alerty oraz czas trwania okna live.
3. Aktualizuj tablicę decyzji w `TradingDecisionJournal` (oznaczenie przejścia paper→live) i zgłoś do compliance.
4. Potwierdź z zespołem SRE, że metryki SLO/SLI nie przekraczają progów (`bot_ui_*`, `risk_service` etc.).

## 5. Referencje

- `docs/ARCHITECTURE.md` – opis procesu demo → paper → live i zależności modułów runtime.
- `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md` – historyczna checklista etapów.
- `docs/runbooks/STAGE6_RESILIENCE_CHECKLIST.md` – procedury awaryjne dla adapterów.
- `docs/runbooks/STAGE6_PORTFOLIO_CHECKLIST.md` – kontrola portfelowa po migracji.
