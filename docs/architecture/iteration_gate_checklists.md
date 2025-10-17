# Checklisty bramek iteracyjnych Etapów 4–5

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

> **Uwaga:** Każdy punkt checklisty wymaga dokumentacji w decision logu podpisanym kluczem HMAC oraz oznaczenia statusu w `stage4_progress.md`. W przypadku regresu (np. nieudany smoke test) pozycje należy przywrócić do `[ ]`, a metryki postępu zaktualizować przed kolejnym podejściem. Na potrzeby Etapu 5 należy dodatkowo odnotować spełnienie wymagań TCO/DecisionOrchestrator zgodnie ze specyfikacją `docs/architecture/stage5_spec.md`.

## 3. Checklista AI Decision Pipeline

| Krok | Opis | Dowód/Artefakt |
| --- | --- | --- |
| 1 | Artefakt modelu (`ModelArtifact`) zapisany w repozytorium modeli z metadanymi `target_scale`, `feature_scalers`, `training_rows`/`validation_rows` i metrykami MAE/RMSE (train + validation). | Plik JSON w katalogu wersji + podpis w decision logu. |
| 2 | Walidacja walk-forward (`WalkForwardValidator`) zakończona sukcesem, średnie MAE/directional accuracy zapisane w raporcie. | Raport z `tests/decision/test_scheduler.py` + log walidacji w `audit/ai_decision/`. |
| 3 | Scheduler retreningu (`RetrainingScheduler`) posiada aktualny `last_run` i zaplanowany `next_run`. | Zrzut konfiguracji/metryk w `audit/ai_decision/scheduler.json`. |
| 4 | `DecisionOrchestrator` zintegrowany z inference (`DecisionModelInference.is_ready == True`), `AIDecisionLoop` generuje kandydatów on-line. | Log z uruchomienia pętli oraz wpis audytowy w decision journal. |
| 5 | Procedury compliance z `docs/architecture/ai_decision_pipeline.md` odhaczone (artefakty, monitoring danych OHLCV, alerty inference). | Checklist podpisana przez Risk/Compliance. |
| 6 | Review dystrybucji rozszerzonych cech trendowych i ryzyka (`ema_*_gap`, `dema_gap`, `tema_gap`, `frama_*`, `rsi`, `stochastic_*`, `stochastic_rsi`, `williams_r`, `cci`, `volume_zscore`, `volatility_trend`, `atr_ratio`, wskaźniki Bollingera, `macd_*`, `ppo_*`, `fisher_transform`, `fisher_signal_gap`, `schaff_trend_cycle`, `trix`, `ultimate_oscillator`, `ease_of_movement`, `vortex_*`, `price_rate_of_change`, `chande_momentum_oscillator`, `detrended_price_oscillator`, `aroon_*`, `balance_of_power`, `relative_vigor_*`, `true_strength_*`, `connors_rsi`, `elder_ray_*`, `ulcer_index`, `efficiency_ratio`, `kama_*`, `qstick`, `obv_normalized`, `pvt_normalized`, pozytywny/negatywny indeks wolumenowy, `di_*`, `adx`, `mfi`, `force_index_normalized`, cechy Ichimoku, `donchian_*`, `chaikin_money_flow`, linia akumulacji/dystrybucji, `chaikin_oscillator`, metryki Heikin Ashi, kanały Keltnera, `vwap_gap`, `psar_*`, pivoty P/R1/S1, luki fraktalne, `coppock_curve`, `choppiness_index`, `intraday_intensity`, `intraday_intensity_volume`, `mass_index`, wskaźnik Klingera (`klinger_*`)) na zbiorach treningowych i walidacyjnych. | Raport z analizy statystyk `feature_stats` dołączony do decision journal. |
