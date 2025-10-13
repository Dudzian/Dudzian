# Raport zamknięcia Etapu 4

## Kontekst
Etap 4 programu rozwoju bota handlowego `bot_core` skupiał się na rozszerzeniu biblioteki strategii o warianty mean reversion, volatility targeting i cross-exchange arbitrage oraz na budowie harmonogramu wielostrate-gicznego, który umożliwia równoległą orkiestrację profili ryzyka w ścieżce demo→paper→live. Dodatkowo wymagano wzmocnienia dokumentacji operacyjnej, harnessu testowego silnika ryzyka oraz procedur audytowych (HMAC, RBAC, mTLS) zgodnych z naszym reżimem compliance. Po audycie zamknięcia zakres został rozszerzony o pakiet obserwowalności, danych, CI i operacji (zadania 7.x–11.x w trackerze) – niniejszy raport dokumentuje ukończone deliverables iteracji 4H.

## Zakres dostarczony
- **Strategie**: komplet ustawień (`Settings`) i klas strategii dla mean reversion, volatility targeting i arbitrażu międzygiełdowego zintegrowanych z warstwą danych, silnikiem ryzyka i modułem egzekucji.
- **Scheduler**: moduł `multi_strategy_scheduler` wraz z CLI i telemetryką, integrujący strategie z runtime oraz pipeline’em demo→paper→live.
- **Konfiguracja**: nowe modele Pydantic, loader presetów i walidatory koszyków instrumentów oraz mapowanie profili ryzyka (konserwatywny, zbalansowany, agresywny, manualny).
- **Testy**: regresja obejmująca strategie, scheduler, ścieżkę runtime oraz scenariusze ryzyka (force liquidation, reset dziennego limitu strat, ograniczenia ekspozycji).
- **Dokumentacja**: specyfikacja Etapu 4, plan testów, checklisty gate’ów, runbook paper tradingu i dedykowane opracowania strategii.
- **Dane backtestowe**: biblioteka znormalizowanych datasetów (manifest + CSV) wraz z walidatorem `DataQualityValidator` i procedurą CLI `validate_backtest_datasets.py`.
- **Automatyzacja CI**: workflow GitHub Actions z zestawem testów strategii/scheduler-a, progiem coverage ≥ 85 % oraz smoke CLI `smoke_demo_strategies.py` bazującym na znormalizowanych datasetach.

## Testy regresyjne
| Obszar | Komenda | Status |
| --- | --- | --- |
| Strategie i scheduler | `PYTHONPATH=. pytest tests/test_mean_reversion_strategy.py tests/test_volatility_target_strategy.py tests/test_cross_exchange_arbitrage_strategy.py tests/test_multi_strategy_scheduler.py` | ✅ |
| Telemetria i journaling | `PYTHONPATH=. pytest tests/test_telemetry_risk_profiles.py tests/test_trading_decision_journal.py` | ✅ |
| Konfiguracja | `PYTHONPATH=. pytest tests/test_core_config_instrument_buckets.py` | ✅ |
| Silnik ryzyka | `PYTHONPATH=. pytest tests/test_risk_engine.py::test_combined_strategy_orders_respect_max_position_pct tests/test_risk_engine.py::test_force_liquidation_due_to_drawdown_allows_only_reducing_orders tests/test_risk_engine.py::test_daily_loss_limit_resets_after_new_trading_day` | ✅ |
| Runtime demo→paper→live | `PYTHONPATH=. pytest tests/test_runtime_pipeline.py` | ✅ |
| Dane backtestowe | `PYTHONPATH=. pytest tests/test_backtest_dataset_library.py`, `python scripts/validate_backtest_datasets.py` | ✅ |
| Smoke CLI demo | `python scripts/smoke_demo_strategies.py --cycles 3` | ✅ |

Pełne logi testów są wersjonowane w katalogu `logs/` wraz z podpisami HMAC generowanymi przez `verify_decision_log.py`.

## Profil ryzyka i compliance
- Zachowany brak wykorzystania WebSocketów – wszystkie integracje korzystają z gRPC/HTTP2 lub IPC.
- Wymuszone RBAC oraz mTLS w kanałach komunikacyjnych runtime’u i modułu scheduler-a, certyfikaty przypięte w konfiguracji.
- Decision log w formacie JSONL podpisywany HMAC, z aktualizacją procedury reagowania na force liquidation i dzienne limity strat.
- Kompatybilność multiplatformowa (Windows/macOS/Linux) utrzymana poprzez abstrakcję ścieżek i brak zależności chmurowych.

## Wnioski i rekomendacje
1. **Stabilność**: Harness ryzyka pokrywa scenariusze kumulacji ekspozycji w trybie multi-strategy – rekomendujemy uruchamianie suite’u regression co najmniej raz dziennie w pipeline’ie CI `run_paper_smoke_ci.py`.
2. **Monitoring**: Dostarczono zintegrowane metryki scheduler-a/strategii oraz reguły Alertmanagera – zalecane jest bieżące monitorowanie paneli Grafany i logów decision logu w celu wychwycenia regresji budżetów ryzyka.
3. **Przejście do Etapu 5**: Możemy rozpocząć prace planistyczne nad etapem dotyczącym optymalizacji kosztów transakcyjnych i rozbudowy decision engine’u, zachowując wymuszone bramki compliance.

## Status iteracji 4AA
- **Postęp Etapu 4**: 40/40 (100 %) – pasek `[####################]`.
- **Ostatnie iteracje utrzymaniowe**:
  - 4I – domknięte zadania bezpieczeństwa (RBAC/mTLS audyt `audit_security_baseline` z obsługą scheduler-a), rozszerzony schemat decision logu, playbook L1/L2, szkolenie operatorów, test obciążeniowy `load_test_scheduler.py` oraz procedury rollbacku.
  - 4K – hotfix obsługi strażników `NONE`/`NULL`/`DEFAULT` w `watch_metrics_stream` bez wymuszania TLS.
  - 4L – wsparcie dla niestandardowych nagłówków gRPC w `watch_metrics_stream` (flagi CLI i zmienne środowiskowe) wraz z sanitacją metadanych decision logu.
  - 4M – wprowadzenie mapowania `grpc_metadata` w `core.yaml`, walidowanego przez loader i walidator konfiguracji oraz honorującego strażniki środowiskowe przy automatycznym wstrzykiwaniu nagłówków do połączeń gRPC telemetryki.
  - 4N – umożliwienie referencji `value_env`/`value_file` w `grpc_metadata`, pobieranie wartości z bezpiecznych źródeł oraz propagacja informacji o pochodzeniu nagłówków do decision logu telemetryki.
  - 4O – poprawa normalizacji wpisów `grpc_metadata`, aby wariant słownikowy korzystający z `value_env`/`value_file` działał zgodnie z oczekiwaniami oraz zachowywał informację o źródle w konfiguracji runtime.
  - 4V – dodanie wariantów `value_base64`/`value_env_base64`/`value_file_base64` w `core.yaml`, co umożliwia wstrzykiwanie nagłówków tekstowych i binarnych dekodowanych z base64 bezpośrednio z centralnej konfiguracji wraz z audytem źródeł.
  - 4W – wsparcie flagi `--headers-file` w `watch_metrics_stream`, które pozwala ładować nagłówki gRPC z plików (obsługując komentarze oraz separatory linii/średników), scalać je z ENV/CLI i raportować źródła w decision logu telemetrycznym.
  - 4P – deduplikacja nagłówków gRPC przy łączeniu konfiguracji `core.yaml` z parametrami CLI, co pozwala operatorowi nadpisywać preset bez dublowania kluczy w wywołaniach gRPC.
  - 4Q – agregacja źródeł nagłówków gRPC (konfiguracja, ENV, CLI) oraz zapis ich pochodzenia w decision logu `watch_metrics_stream`, aby audyt mógł prześledzić końcową konfigurację telemetryki.
  - 4R – wsparcie dyrektyw `@env:`/`@file:` w CLI `--header`, pozwalające bezpiecznie wstrzykiwać wartości z ENV/plików i eskalować źródła do decision logu wraz z escape `@@` dla literalnych wartości.
  - 4S – uelastycznienie zmiennej `BOT_CORE_WATCH_METRICS_HEADERS` o separator linii i średniki, co umożliwia wygodne definiowanie nagłówków w plikach `.env`/zmiennych środowiskowych bez dodatkowego escapowania.
  - 4U – wsparcie dla referencji `@env64:`/`@file64:` oraz automatycznej dekodacji base64 (także dla nagłówków binarnych), co umożliwia bezpieczne przechowywanie sekretów metadanych poza repozytorium i wstrzykiwanie ich w czasie uruchomienia.
  - 4T – pomijanie linii komentarza rozpoczynających się od `#` w `BOT_CORE_WATCH_METRICS_HEADERS`, dzięki czemu można przechowywać nagłówki w plikach `.env` wraz z komentarzami dokumentującymi przeznaczenie poszczególnych wpisów.
  - 4X – dodanie zmiennej środowiskowej `BOT_CORE_WATCH_METRICS_HEADERS_FILE`, obsługującej listę plików z nagłówkami gRPC (z komentarzami i separatorami linii/średników) oraz ich scalanie z konfiguracją `core.yaml`, wpisami CLI i zmienną `HEADERS` przy zachowaniu audytu źródeł.
  - 4Y – rozszerzenie `core.yaml` o pole `grpc_metadata_files`, które pozwala centralnie wskazać pliki z nagłówkami gRPC. Loader normalizuje ścieżki względem katalogu konfiguracji, a `watch_metrics_stream` wczytuje wpisy przed presetami `grpc_metadata`, respektując strażników `NONE/NULL` i raportując źródła oraz usunięcia w decision logu.
  - 4Z – wsparcie katalogów z nagłówkami gRPC poprzez pola `grpc_metadata_directories`, flagę `--headers-dir` oraz zmienne `BOT_CORE_WATCH_METRICS_HEADERS_DIRS/FILE`, co umożliwia operacyjne utrzymywanie zestawów plików, ich deterministyczne scalanie i audyt pełnej mapy źródeł w decision logu.
  - 4AA – raport `--headers-report`/`--headers-report-only` w `watch_metrics_stream`, umożliwiający wypisanie scalonych nagłówków gRPC (klucz, typ, źródło, usunięcia) z maskowaniem wrażliwych wartości i bez konieczności zestawiania kanału gRPC.
- **Czynności otwarte**: brak – backlog Etapu 4 został zrealizowany w całości.
- **Blokery**: brak nowych ryzyk; budżety zasobów monitorowane przez `runtime.resource_limits` i `resource_monitor` mieszczą się w zadeklarowanych progach.

