# Runtime UI

PySide6 klient korzysta z `ui/backend/runtime_service.py`, aby udostępnić QML
aktualny stan decyzji AI, blokad ryzyka, alertów i kanałów gRPC. Serwis
kapsułkuje komunikację z `bot_core.runtime` oraz zapewnia sygnały Qt, które można
podpinać w widokach blur/FontAwesome.

## RuntimeService – najważniejsze funkcje

* `refreshDecisions()` i `decisionsChanged` pobierają najnowsze decyzje AI z
  backendu i udostępniają je panelowi Explainability oraz kreatorom trybów.
* `requestUnblock()` i `requestFreeze()` umożliwiają operatorowi zarządzanie
  blokadami/override’ami bezpośrednio z UI – kontroler samodzielnie loguje
  akcje oraz odświeża stan modeli.【F:ui/backend/runtime_service.py†L1458-L1625】
* `activitySummary()` normalizuje liczbę blokad, zamrożeń i override’ów dla
  poszczególnych profili i udostępnia dane w strukturze przyjaznej QML. Widok
  ryzyka może dzięki temu kolorować karty zgodnie z aktywnością profilu.【F:ui/backend/runtime_service.py†L385-L570】
* `RuntimeService` emituje sygnały `alertsChanged`, `riskStateChanged`,
  `licensingChanged` i `cloudRuntimeStatusChanged`, co upraszcza spójne
  aktualizacje paneli statusu i powiadomień.

## Telemetria i pojedyncza instancja

UI przekazuje naruszenia polityki single-instance do backendu poprzez
`bot_core.security.fingerprint.report_single_instance_event`. Helper dostępny w
CLI (`python -m bot_core.security.fingerprint report-single-instance ...`)
wysyła zdarzenie `ui_single_instance_conflict`, które trafia do
`logs/security_admin.log`. Dzięki temu operatorzy wiedzą, kiedy druga kopia UI
próbuje wystartować na tym samym hostcie.【F:bot_core/security/fingerprint.py†L1439-L1497】

## Integracja z testami

* `tests/ui/test_single_instance.py` sprawdza, że funkcja
  `report_single_instance_event` zapisuje poprawny payload i jest wywoływana z
  CLI.
* `tests/ui_pyside/test_app_bootstrap.py` startuje PySide6 w trybie offscreen i
  potwierdza, że `RuntimeService` oraz pozostałe kontrolery zostały poprawnie
  zarejestrowane w QML.
