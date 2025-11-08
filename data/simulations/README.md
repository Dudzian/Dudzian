# Sandbox backtestowy strategii

Repozytorium zawiera zestaw scenariuszy walidacyjnych dla nowych strategii
marketplace. Konfiguracje znajdują się w plikach `*.yaml` i są wykorzystywane
przez `StrategyQualityPipeline` do generowania raportów w katalogu
`reports/strategies`.

Każdy scenariusz definiuje:

* `engine` – identyfikator silnika z `DEFAULT_STRATEGY_CATALOG`.
* `parameters` – konfigurację przekazywaną do strategii podczas symulacji.
* `calibration_metrics` – wyniki historycznego backtestu sandboxowego.
* `acceptance_criteria` – minimalne/maksymalne wartości metryk akceptacyjnych.
* `notes` – dodatkowe komentarze dla zespołu review.

Aby uruchomić walidację jakości:

```bash
python -m bot_core.strategies.quality_pipeline
```

Moduł załaduje wszystkie scenariusze z katalogu, oceni je względem progów i
zapisze raporty JSON wraz z podsumowaniem.

