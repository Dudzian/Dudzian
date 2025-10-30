# Raporty z optymalizacji strategii

Moduł `bot_core.reporting.optimization` umożliwia generowanie raportów HTML, JSON oraz prostego PDF
na podstawie obiektu `StrategyOptimizationReport`. Raporty zawierają metadane zadania (silnik,
algorytm, cel optymalizacji), listę wszystkich prób wraz z oceną oraz wyróżnienie najlepszego zestawu
parametrów.

## Szybki przykład

```python
from pathlib import Path
from bot_core.optimization import StrategyOptimizer
from bot_core.reporting.optimization import export_report

# ... po wykonaniu optymalizacji ...
report = optimizer.optimize(...)
export_paths = export_report(report, Path("var/reports/optimization"), formats=("html", "json", "pdf"))
print("Zapisano:", export_paths)
```

Każdy raport jest zapisywany z nazwą `<strategia>_<timestamp>.<format>` i może być wczytany przez UI
lub udostępniony użytkownikowi końcowemu.
