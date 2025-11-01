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


## Raport retreningu

Raport generowany przez `scripts/run_retraining_cycle.py` po zakończeniu pojedynczego cyklu retreningu. Zawiera KPI (czas trwania, opóźnienia, dryf), łańcuch fallbacków backendów ML, zarejestrowane zdarzenia oraz alerty chaosu. Artefakty Markdown i JSON trafiają do `reports/retraining/` i mogą być archiwizowane w pipeline CI.

Scenariusz E2E walidujący retrening (`pytest -m e2e_retraining`) wykorzystuje te same raporty, dodatkowo zapisując snapshot KPI oraz log przebiegu w katalogach `reports/e2e/retraining/` i `logs/e2e/retraining/`. Ułatwia to manualne porównanie wyników i dystrybucję artefaktów do zespołów QA.
