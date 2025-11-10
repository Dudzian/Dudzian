# Stage6 coverage for removed legacy paths

Poniższa tabela dokumentuje usunięte ścieżki `legacy` oraz odpowiadające im testy regresyjne.

| Obszar | Nowe API Stage6 | Testy pokrywające |
| --- | --- | --- |
| `bot_core/runtime/pipeline.DecisionAwareSignalSink` | `evaluation_summary_v2()` jest jedynym publicznym API Stage6; usunęliśmy `evaluation_summary()` oraz `_legacy_evaluation_summary()` | `tests/runtime/test_decision_orchestrator_runtime.py::test_decision_orchestrator_runtime_creates_summary`, `tests/runtime/test_streaming_feed.py::test_streaming_feed_collects_summary`
| `bot_core/runtime/metrics_service.build_metrics_server` | Metadane `shared_secret_token` zastępują `legacy_token` w raportach | `tests/test_audit_service_tokens_script.py::test_audit_service_tokens_script_warns_on_shared_secret`, `tests/test_security_token_audit.py::test_audit_warns_when_env_missing_and_shared_secret_only`
| `bot_core/trading/auto_trade.TradingEngine` | `_stage6_signal_bundle()` zastępuje `_legacy_signal_bundle()` jako pakiet bazowy sygnałów | `tests/test_trading_engine_native.py::TestNativeTradingEngine.test_signal_and_risk_services`
| `bot_core/ai/models.ModelMetrics` | Stage6 wymaga bloków `summary/train/validation/test`; odrzucono spłaszczone payloady legacy | `tests/ai/test_models_exports.py::test_model_metrics_rejects_legacy_payload`
| `ui/src/app/Application::loadUiSettings` | Ostrzeżenia wymuszają format Stage6 i sygnalizują brak migracji do legacy | `tests/test_security_ui_bridge.py::test_dump_state_rejects_legacy_bundle_format`, `tests/test_security_ui_bridge.py::test_dump_state_reports_invalid_on_corrupted_signature` (oba scenariusze potwierdzają brak fallbacku)
| `scripts/run_stress_lab.py` | Stage6 CLI wymaga jawnej subkomendy (`evaluate` lub `run`) – brak fallbacku na stare ścieżki | `tests/test_stress_lab_script.py::test_runbook_evaluate_command`, `tests/test_stress_lab_script.py::test_runbook_run_command`

Każdy wpis potwierdza, że nowe ścieżki Stage6 posiadają regresję testową, a pozostałości `legacy` zostały zablokowane przez lint `scripts/lint_paths.py`.
