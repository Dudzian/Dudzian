# Market Intelligence Stage6

Ten katalog nie zawiera już binarnego dumpu SQLite. Aby odtworzyć przykładowe dane do demonstracji Stage6, uruchom:

```
python scripts/build_market_intel_metrics.py \
  --config config/core.yaml \
  --sqlite-path var/stage6/market_intel/market_metrics.sqlite \
  --output-dir data/stage6/metrics \
  --manifest var/audit/stage6/market_intel/manifest.json
```

Przed wykonaniem polecenia przygotuj bazę SQLite zgodnie z własnym pipeline'em (np. `sqlite3 var/stage6/market_intel/market_metrics.sqlite < seed.sql>`).
