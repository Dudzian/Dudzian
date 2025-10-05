# Próbka manifestu coverage

Ten katalog zawiera minimalną konfigurację `core.yaml`, która służy jako fixture
integracyjny dla runnera pokrycia danych (coverage alert runner). Dane manifestu
SQLite są generowane na żądanie za pomocą skryptu `scripts/build_sample_manifest.py`
i obejmują scenariusz smoke — dwa instrumenty z kompletnym pokryciem D1 do
2024-01-30.

Konfiguracja zawiera również domyślne progi jakości danych przypisane do profilu
ryzyka `balanced` (`max_gap_minutes=2160`, `min_ok_ratio=0.9`). Fallback
zapewnia, że środowiska bez własnych limitów `data_quality` zachowują wymóg
aktualizacji świec co najwyżej co 36 godzin przy utrzymaniu co najmniej 90% wpisów
ze statusem OK.

Regeneracja manifestu:

```bash
python scripts/build_sample_manifest.py --output-dir tests/assets/coverage_sample
```

Skrypt usuwa konieczność przechowywania plików binarnych w repozytorium, a dane
pozostają w pełni zanonimizowane.
