# Integracja runbooków operacyjnych z UI

Panel **RunbookPanel** w aplikacji desktopowej udostępnia operatorom listę
alertów guardrail’i wraz z przypisanymi runbookami. Dane są synchronizowane z
raportem generowanym przez `core.reporting.guardrails_reporter.GuardrailReport`.

## Jak działa mapowanie

1. `RunbookController` odświeża raport guardrail’i i przekształca go w listę
   alertów (metryki, logi i rekomendacje).
2. Każdy alert jest dopasowywany do runbooka na podstawie słów kluczowych:
   - timeouty i błędy sieci → `strategy_incident_playbook.md`
   - oczekiwanie na limity/rate limit → `autotrade_threshold_calibration.md`
   - problemy licencyjne/OEM → `oem_license_provisioning.md`
3. Operator może wyświetlić szczegóły i otworzyć runbook bezpośrednio z UI.

## Dodawanie nowych runbooków

1. Dodaj plik `*.md` do katalogu `docs/operations/runbooks/` z nagłówkiem `#` na
   początku – nagłówek będzie użyty jako tytuł w UI.
2. Dodaj plik metadanych YAML w `docs/operations/runbooks/metadata/` o tej samej
   nazwie (np. `example_runbook.yml`). Metadane opisują manualne kroki oraz
   opcjonalne akcje automatyczne:

   ```yaml
   id: example_runbook
   automatic_actions:
     - id: restart_service
       label: "Uruchom ponownie usługę"
       script: restart_service.py
       description: "Wywołuje skrypt w katalogu scripts/runbook_actions/."
       confirm: "Czy chcesz kontynuować restart?"
   manual_steps:
     - "Sprawdź status usługi w panelu telemetry."
   ```

3. Umieść skrypt automatyczny w `scripts/runbook_actions/` (Python uruchamiany
   lokalnie) i zapewnij, że zwraca kod wyjścia `0` przy powodzeniu. Wszystkie
   skrypty logują przebieg do `logs/ui/runbook_actions/`.
4. Zaktualizuj mapowanie słów kluczowych w `ui/backend/runbook_controller.py`
   (`_KEYWORD_MAP`), aby alerty guardrail były poprawnie kierowane.
5. Uzupełnij dokumentację, jeżeli runbook wymaga dodatkowego kontekstu lub
   procedur bezpieczeństwa.

Po zapisaniu zmian panel UI automatycznie wykryje nowy runbook, wyświetli kroki
manualne, a operator otrzyma możliwość uruchomienia akcji automatycznych z
poziomu panelu wraz z potwierdzeniem bezpieczeństwa.

## Bezpieczeństwo akcji automatycznych

- Każda akcja powinna wymagać potwierdzenia, jeśli może powodować przerwę w
  działaniu bota lub wpływać na licencjonowanie. Wykorzystaj klucz `confirm` w
  metadanych.
- Skrypty w `scripts/runbook_actions/` działają lokalnie i powinny być
  idempotentne – kolejne uruchomienie nie powinno pogarszać sytuacji.
- Wszystkie operacje są logowane w `logs/ui/runbook_actions/`. Operatorzy mogą
  dołączyć logi do paczki diagnostycznej (`scripts/generate_diagnostics.py`).
- Dodając nowe akcje, uwzględnij ewentualne zależności systemowe i opisz je w
  dokumentacji runbooka, aby uniknąć uruchamiania skryptów w nieodpowiednich
  warunkach.
