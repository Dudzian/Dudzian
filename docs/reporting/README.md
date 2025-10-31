# Raportowanie OEM

Ten katalog opisuje raporty generowane przez narzędzia wspierające wdrożenia lokalnego runtime'u.

## Raport scenariusza demo → paper

Raport tworzony przez `scripts/run_local_bot.py` po zakończeniu scenariusza E2E. Pliki Markdown trafiają do `reports/e2e/` i zawierają status kroków, KPI oraz ostrzeżenia/błędy.

## Raport guardrail'i kolejki I/O

Nowy moduł `core.reporting.guardrails_reporter.GuardrailReport` gromadzi metryki z rejestru Prometheusa (`exchange_io_*`) oraz logi `logs/guardrails/events.log`. Raport jest zapisywany w `reports/guardrails/` i dołączany do artefaktów scenariusza demo → paper.

W raporcie znajdują się:

- tabela z agregacją timeoutów i oczekiwania na limity (`AsyncIOTaskQueue`) per środowisko i kolejkę,
- log z ostatnimi zdarzeniami guardrail'i,
- rekomendacje dotyczące konfiguracji limitów i diagnostyki błędów.

Raport generuje się automatycznie z poziomu `scripts/run_local_bot.py` i jest również eksportowany do pliku JSON w ramach głównego raportu E2E (sekcja `guardrails`).
