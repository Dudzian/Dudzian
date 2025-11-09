# Market Intelligence Stage6

Ten katalog nie zawiera już binarnego dumpu SQLite. Aby odtworzyć przykładowe dane do demonstracji Stage6, skorzystaj z buildera,
który automatycznie zasili bazę `market_metrics.sqlite` i wyeksportuje JSON-y:

```
python scripts/build_market_intel_metrics.py \
  --config config/core.yaml \
  --sqlite-path var/stage6/market_intel/market_metrics.sqlite \
  --output-dir data/stage6/metrics \
  --manifest var/audit/stage6/market_intel/manifest.json \
  --required-symbol BTCUSDT --required-symbol ETHUSDT \
  --populate-sqlite \
  --sqlite-provider stage6_samples.market_intel:build_provider
```

Flagę `--sqlite-provider` wskazującą moduł `module:callable` należy zastąpić własnym providerem, który wykorzystuje adaptery
`bot_core.exchanges/*` lub pipeline danych (`bot_core/data/*`). Builder zweryfikuje sumy kontrolne po zapisie i przerwie działanie,
jeśli baza nie zawiera kompletnego zestawu metryk. Dotychczasowy manualny krok `sqlite3 ... < seed.sql>` nie jest już potrzebny –
seedowanie odbywa się poprzez provider.
