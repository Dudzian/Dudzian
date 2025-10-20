# Market regime calibration

Kalibracja progów `MarketRegimeClassifier` została potwierdzona na reprezentatywnych próbkach OHLCV w katalogu
`data/sample_ohlcv`. Dla każdej próbki uruchamiany jest pełen pipeline `MarketRegimeClassifier` → `RegimeHistory`,
co opisują testy `tests/test_sample_ohlcv_regimes.py`. Wygenerowane streszczenia są porównywane z oczekiwanym
reżimem, poziomem ryzyka oraz zakresem wyniku ryzyka. Odchylenia są raportowane przez logger `tests.sample_data`.

Zestawienie domyślnych progów zapisano w `config/regime_thresholds.yaml`. Test regresyjny
`tests/test_regime_thresholds_config.py` chroni przed przypadkowym powrotem do niekalibrowanych wartości.
