# Próbka manifestu coverage

Ten katalog zawiera minimalną konfigurację `core.yaml`, która służy jako fixture
integracyjny dla runnera pokrycia danych (coverage alert runner). Dane manifestu
SQLite są generowane na żądanie za pomocą skryptu `scripts/build_sample_manifest.py`
i obejmują scenariusz smoke — dwa instrumenty z kompletnym pokryciem D1 do
2024-01-30.

Konfiguracja zawiera również domyślne progi jakości danych przypisane do profilu
ryzyka `balanced`, które są wykorzystywane jako fallback, gdy środowisko nie
definiuje własnych limitów `data_quality`.

Regeneracja manifestu:

```bash
python scripts/build_sample_manifest.py --output-dir tests/assets/coverage_sample
```

Skrypt usuwa konieczność przechowywania plików binarnych w repozytorium, a dane
pozostają w pełni zanonimizowane.
