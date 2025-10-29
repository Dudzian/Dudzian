# Strategia ML + LLM

Dokument opisuje sposób wdrożenia hybrydowych strategii łączących uczenie maszynowe oraz modele językowe.

## Architektura

1. **Warstwa cech** – `bot_core.data.pipelines.ml_features.MLFeaturePipeline` odpowiada za budowę znormalizowanych cech i targetów na potrzeby modeli klasycznych oraz sekwencyjnych.
2. **Adaptery modeli** – katalog `bot_core/strategies/ml/` zawiera klasy bazowe oraz adaptery dla RandomForest, XGBoost, LSTM i Temporal Fusion Transformer. Pozwalają one ujednolicić API `fit/predict` niezależnie od użytej biblioteki.
3. **Silnik strategii** – `MLStrategyEngine` integruje pipeline cech, adapter modelu i logikę generowania sygnałów. Wspiera próg decyzyjny oraz serializację konfiguracji.
4. **Interfejs LLM** – `bot_core.ai.llm_strategy_adapter.LLMStrategyAdapter` odpowiada za przygotowanie promptów, wywołania modeli językowych oraz lokalne narzędzia (analiza sentymentu, interpretacja newsów).

## Uczenie i walidacja

Parametry uczenia znajdują się w `config/strategies/ml.yaml`. Zawierają one konfigurację pipeline'u danych, zestaw modeli oraz ustawienia walidacji walk-forward (okna treningowe, kroki przesuwu, metryki). Pipeline zapewnia generowanie zbiorów treningowych i normalizację, co ułatwia integrację z adapterami modeli.

## Testy i benchmarki

Działanie pipeline'u, silnika strategii oraz adapterów jest weryfikowane w `tests/strategies/ml/`. Testy regresyjne obejmują budowę cech, generowanie sygnałów i zachowanie adapterów. W razie potrzeby można rozszerzyć je o benchmarki czasowe, bazując na tych samych fixture'ach.

## Integracja produkcyjna

* **Monitorowanie** – rekomendowane jest logowanie metadanych sygnałów (prognoza, threshold) oraz wersji modelu.
* **Aktualizacja modeli** – adaptery udostępniają metodę `save`, dzięki czemu można przechowywać konfiguracje i metadane w repozytorium artefaktów.
* **Rozszerzenia** – interfejs LLM wspiera dodawanie własnych narzędzi, np. klasyfikacji newsów branżowych lub generowania komentarzy do sygnałów.
