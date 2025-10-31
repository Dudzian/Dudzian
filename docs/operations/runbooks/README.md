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

1. Dodaj plik `*.md` do katalogu `docs/runbooks/operations/` z nagłówkiem `#` na
   początku – nagłówek będzie użyty jako tytuł w UI.
2. Zaktualizuj mapowanie słów kluczowych w
   `ui/backend/runbook_controller.py` (`_KEYWORD_MAP`).
3. Opcjonalnie rozszerz dokumentację, jeśli runbook wymaga dodatkowego
   kontekstu.

Po zapisaniu zmian panel UI automatycznie wykryje nowy runbook i umożliwi jego
wyświetlenie w odpowiedzi na pasujące alerty guardrail’i.
