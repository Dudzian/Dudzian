# Checklisty bramek iteracyjnych Etapu 4

Dokument zbiera rozszerzone checklisty wejścia i wyjścia dla iteracji rozwojowych Etapu 4. Wszystkie punkty odnoszą się do pipeline'u demo → paper → live i muszą być odhaczone przed przejściem do kolejnej fazy. Kryteria integrują smoke testy środowiska paper, audyty decyzji oraz zgodność z profilami ryzyka.

## 1. Checklista wejścia iteracji

| Krok | Opis | Dowód/Artefakt |
| --- | --- | --- |
| 1 | Aktualizacja mapy profili ryzyka oraz koszyków instrumentów w `config/core.yaml`; uruchom `python scripts/validate_config.py --config config/core.yaml` | Raport walidacji zapisany w `audit/config_validation/iteracja_<nr>.json` |
| 2 | Synchronizacja presetów strategii w repo (`MeanReversionSettings`, `VolatilityTargetSettings`, `CrossExchangeArbitrageSettings`) z dokumentacją techniczną | PR z linkiem do `docs/architecture/strategies/*.md` |
| 3 | Weryfikacja smoke testu paper: `PYTHONPATH=. python scripts/run_paper_smoke_ci.py --environment binance_paper --render-summary-markdown` | `audit/paper_smoke/summary.json` + podpis HMAC w `docs/audit/paper_trading_log.jsonl` |
| 4 | Audyt tokenów RBAC i certyfikatów mTLS (`python scripts/audit_service_tokens.py --config config/core.yaml`, `python scripts/audit_tls_assets.py --config config/core.yaml`) | Raporty JSON w `audit/rbac/` i `audit/tls/` z aktualnymi SHA-256 |
| 5 | Aktualizacja checklist operacyjnych i planu testów regresyjnych (`docs/architecture/stage4_test_plan.md`, `docs/runbooks/paper_trading.md`) | Commit z referencją do numeru iteracji w `docs/architecture/stage4_progress.md` |

## 2. Checklista wyjścia iteracji

| Krok | Opis | Dowód/Artefakt |
| --- | --- | --- |
| 1 | Wyniki regresji jednostkowych i integracyjnych (`pytest tests/test_mean_reversion_strategy.py tests/test_multi_strategy_scheduler.py ...`) | Log z CI lub `logs/tests/iteracja_<nr>.txt` |
| 2 | Udane smoke testy paper (`summary.json`, `paper_smoke_report.zip`) zsynchronizowane przez `publish_paper_smoke_artifacts.py` oraz podpisane HMAC (`verify_decision_log.py`) | Raport `publish_summary.json` + wpis w `audit/paper_trading_log.md` |
| 3 | Decision log JSONL zawiera podpisane decyzje dla wszystkich strategii aktywnych w schedulerze (`python scripts/verify_decision_log.py --log audit/decisions --hmac-key $DECISION_KEY`) | `audit/decisions/verification_report.json` |
| 4 | Risk engine raportuje alokacje zgodne z profilami (`python bot_core/runtime/telemetry_risk_profiles.py --summary core`) | Plik `audit/risk_profiles/core_iteracja_<nr>.json` |
| 5 | Review operacyjny i bezpieczeństwa potwierdzony w `docs/audit/paper_trading_log.md` wraz z podpisem operatora | Nowy wpis z datą i identyfikatorem operatora |
| 6 | Zatwierdzona aktualizacja `docs/architecture/stage4_progress.md` oraz `iteration_gate_checklists.md` z procentami postępu | Merge request + notatka w decision logu |

> **Uwaga:** Każdy punkt checklisty wymaga dokumentacji w decision logu podpisanym kluczem HMAC oraz oznaczenia statusu w `stage4_progress.md`. W przypadku regresu (np. nieudany smoke test) pozycje należy przywrócić do `[ ]`, a metryki postępu zaktualizować przed kolejnym podejściem.
