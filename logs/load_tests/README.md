# Artefakty testów obciążeniowych scheduler-a

W katalogu znajdują się przykładowe raporty generowane przez
`scripts/load_test_scheduler.py`.  Plik
`scheduler_profile.sample.json` przedstawia strukturę JSON zapisywaną
po każdorazowym uruchomieniu benchmarku wraz z parametrami wejściowymi.

Aby odtworzyć raport na własnej maszynie:

```bash
PYTHONPATH=. python scripts/load_test_scheduler.py \
  --iterations 5 \
  --schedules 3 \
  --signals 4 \
  --latency-ms 1.0 \
  --output logs/load_tests/scheduler_profile.sample.json
```

Przed publikacją wyników do audytu dołącz również surowe logi z
telemetrii scheduler-a oraz powiązane metryki budżetu zasobów.
